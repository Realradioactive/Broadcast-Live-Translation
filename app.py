import sys
import os

# OMP ÇATIŞMASINI KÖKÜNDEN ÇÖZEN SATIR (PyTorch ve Whisper çakışmasını engeller)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import queue
import json
import time
import warnings
from datetime import datetime, timedelta
import numpy as np

# Gereksiz uyarıları sustur
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import speech_recognition as sr
from deep_translator import GoogleTranslator, MyMemoryTranslator, DeeplTranslator
from transformers import MarianMTModel, MarianTokenizer
import torch
from faster_whisper import WhisperModel

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QCheckBox, QLineEdit, QFileDialog,
                             QTabWidget, QSlider, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from flask import Flask, render_template_string, jsonify
import logging

# ==========================================
# FLASK WEB SUNUCUSU
# ==========================================
flask_app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

current_data = {"recognized": "", "translated": ""}

@flask_app.route('/')
def index():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Canlı Çeviri Yayıncı Ekranı</title>
        <style>
          body { background-color: #121212; color: #00ffcc; font-size: 36px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin-top: 15%; font-weight: bold; text-shadow: 2px 2px 4px #000000;}
          .trans { color: #ffffff; margin-top: 20px; font-size: 42px;}
        </style>
      </head>
      <body>
        <div id="recognized_text">...</div>
        <div id="translated_text" class="trans">...</div>
        <script>
          setInterval(() => {
            fetch('/data').then(res => res.json()).then(data => {
                document.getElementById('recognized_text').innerText = data.recognized;
                document.getElementById('translated_text').innerText = data.translated;
            });
          }, 500);
        </script>
      </body>
    </html>
    ''')

@flask_app.route('/data')
def get_data():
    return jsonify(current_data)

def run_flask():
    flask_app.run(host='127.0.0.1', port=3333, use_reloader=False)

# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def format_srt_time(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

LANG_MAP_GOOGLE = {
    'tr': 'tr-TR', 'en': 'en-US', 'de': 'de-DE', 
    'fr': 'fr-FR', 'es': 'es-ES', 'it': 'it-IT',
    'ru': 'ru-RU', 'ja': 'ja-JP'
}

# ==========================================
# ÇEVİRİ MOTORU WORKER (Whisper GPU Destekli)
# ==========================================
class TranslationWorker(QThread):
    update_text = pyqtSignal(str, str)
    status_update = pyqtSignal(str)

    def __init__(self, stt_mode, whisper_size, trans_engine, api_key, src_lang, dest_lang, 
                 pause_thresh, srt_enabled, srt_path, blacklist_str, vad_ms):
        super().__init__()
        self.stt_mode = stt_mode
        self.whisper_size = whisper_size
        self.trans_engine = trans_engine
        self.api_key = api_key
        self.src_lang = src_lang
        self.dest_lang = dest_lang
        self.pause_thresh = pause_thresh
        self.srt_enabled = srt_enabled
        self.srt_path = srt_path
        
        # Kullanıcının arayüzden girdiği karalisteyi listeye çeviriyoruz
        self.blacklist = [x.strip().lower() for x in blacklist_str.split(",") if x.strip()]
        self.vad_ms = vad_ms
        
        self.srt_index = 1
        self.session_start_time = time.time()
        self.running = True
        self.task_queue = queue.Queue()
        
        self.local_translator = None
        self.local_tokenizer = None
        self.whisper_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def write_srt(self, start_sec, end_sec, text_tr, text_en):
        if not self.srt_enabled or not self.srt_path: return
        start_str = format_srt_time(start_sec)
        end_str = format_srt_time(end_sec)
        try:
            with open(self.srt_path, "a", encoding="utf-8") as f:
                f.write(f"{self.srt_index}\n{start_str} --> {end_str}\n{text_tr}\n{text_en}\n\n")
            self.srt_index += 1
        except Exception as e:
            print(f"SRT Hatası: {e}")

    def audio_listener(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = self.pause_thresh
        recognizer.non_speaking_duration = 0.4
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.session_start_time = time.time()
            self.status_update.emit(f"🟢 Dinleniyor | STT: {self.stt_mode.split(' ')[0]} | Çeviri: {self.trans_engine.split(' ')[0]}")
            
            while self.running:
                try:
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=15)
                    et = time.time() - self.session_start_time
                    st = max(0, et - (len(audio.frame_data) / (audio.sample_rate * audio.sample_width)))
                    self.task_queue.put(("audio_task", audio, st, et))
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    pass

    def run(self):
        if self.srt_enabled:
            with open(self.srt_path, "w", encoding="utf-8") as f: f.write("") 

        if self.trans_engine == "Offline (Lokal Helsinki-NLP)":
            self.status_update.emit(f"⏳ Çeviri Modeli (Helsinki) Yükleniyor [{self.device.upper()}]...")
            try:
                hf_model = 'Helsinki-NLP/opus-mt-tr-en'
                self.local_tokenizer = MarianTokenizer.from_pretrained(hf_model)
                self.local_translator = MarianMTModel.from_pretrained(hf_model).to(self.device)
            except Exception as e:
                self.status_update.emit(f"🔴 Lokal Çeviri Hatası: Ağ bağlantısı veya Model Eksik.")
                return

        if self.stt_mode == "Offline (Lokal Whisper)":
            self.status_update.emit(f"⏳ Ses Modeli (Whisper {self.whisper_size}) GPU'ya Yükleniyor...")
            try:
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.whisper_model = WhisperModel(self.whisper_size, device=self.device, compute_type=compute_type)
            except Exception as e:
                self.status_update.emit(f"🔴 Whisper Yükleme Hatası: {e}")
                return

        listener_thread = threading.Thread(target=self.audio_listener, daemon=True)
        listener_thread.start()

        stt_lang_google = LANG_MAP_GOOGLE.get(self.src_lang, "en-US")
        
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.5)
                if task[0] == "STOP":
                    break
                    
                payload_audio, st, et = task[1], task[2], task[3]
                text = ""
                
                # A. SESİ METNE ÇEVİR (STT)
                if self.stt_mode == "Offline (Lokal Whisper)":
                    try:
                       # MÜDAHALE: Mikrofonun 48kHz olan sesini Whisper'ın zorunlu kıldığı 16kHz'e downsample yapıyoruz!
                        resampled_bytes = payload_audio.get_raw_data(convert_rate=16000, convert_width=2)
                        audio_np = np.frombuffer(resampled_bytes, np.int16).flatten().astype(np.float32) / 32768.0
                        
                        # GUI'den Gelen VAD Parametresi ve Döngü Kırıcı (condition_on_previous_text=False) kullanılıyor
                        segments, _ = self.whisper_model.transcribe(
                            audio_np, 
                            language=self.src_lang, 
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=self.vad_ms),
                            condition_on_previous_text=False
                        )
                        text = " ".join([segment.text for segment in segments]).strip()
                        
                        # ZEHİRLİ KELİME FİLTRESİ (Blacklist)
                        if any(zehir in text.lower() for zehir in self.blacklist):
                            text = "" # Metni iptal et
                            
                    except Exception as e:
                        print(f"Whisper Hatası: {e}")
                        self.task_queue.task_done()
                        continue
                else:
                    try:
                        text = sr.Recognizer().recognize_google(payload_audio, language=stt_lang_google)
                    except Exception:
                        self.task_queue.task_done()
                        continue

                if not text:
                    self.task_queue.task_done()
                    continue

                # B. METNİ ÇEVİR
                translated = ""
                try:
                    if self.trans_engine == "Offline (Lokal Helsinki-NLP)":
                        inputs = self.local_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                        translated_tokens = self.local_translator.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
                        translated = self.local_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    elif self.trans_engine == "Google (Ücretsiz)":
                        translated = GoogleTranslator(source=self.src_lang, target=self.dest_lang).translate(text)
                    elif self.trans_engine == "DeepL (API Key)":
                        if not self.api_key.strip(): translated = "[API Key Yok]"
                        else: translated = DeeplTranslator(api_key=self.api_key.strip(), source=self.src_lang, target=self.dest_lang).translate(text)
                except Exception as e:
                    translated = "[Çeviri Motoru Hatası]"
                
                # C. YAZDIR
                self.write_srt(st, et, text, translated)
                current_data["recognized"] = text
                current_data["translated"] = translated
                self.update_text.emit(text, translated)
                
                self.task_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.task_queue.put(("STOP", None, 0, 0))
        self.quit()
        self.wait() 

# ==========================================
# ŞEFFAF MASAÜSTÜ ALTYAZI EKRANI (OSD)
# ==========================================
class OSDWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.current_opacity = 180

        layout = QVBoxLayout()
        self.lbl_rec = QLabel("Altyazı sistemi hazır...")
        self.lbl_rec.setAlignment(Qt.AlignCenter)
        self.lbl_trans = QLabel("...")
        self.lbl_trans.setAlignment(Qt.AlignCenter)
        
        self.set_styles(24)
        layout.addWidget(self.lbl_rec)
        layout.addWidget(self.lbl_trans)
        self.setLayout(layout)

    def set_styles(self, font_size):
        bg_color = f"rgba(20, 20, 20, {self.current_opacity})"
        self.lbl_rec.setStyleSheet(f"color: #00ffcc; font-size: {font_size}px; font-weight: bold; background-color: {bg_color}; padding: 10px; border-radius: 10px;")
        self.lbl_trans.setStyleSheet(f"color: #ffffff; font-size: {font_size + 8}px; font-weight: bold; background-color: {bg_color}; padding: 10px; border-radius: 10px; margin-top: 5px;")

    def set_opacity(self, value):
        self.current_opacity = int((value / 100) * 255)
        self.set_styles(24)

    def apply_positioning(self, screen_index, position):
        screens = QApplication.screens()
        if screen_index >= len(screens): screen_index = 0
        geom = screens[screen_index].geometry()
        width = int(geom.width() * 0.8)
        height = 200
        x = geom.x() + int((geom.width() - width) / 2)
        y = geom.y() + geom.height() - height - 50 if position == "Alt (Bottom)" else geom.y() + 50 
        self.setGeometry(x, y, width, height)

    def update_text(self, rec, trans):
        self.lbl_rec.setText(rec)
        self.lbl_trans.setText(trans)

# ==========================================
# PROFESYONEL KONTROL PANELİ V8.2
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Çeviri Merkezi V8.2 (Tam Filtreli .Exe Sürümü)")
        self.resize(680, 680)
        self.worker = None
        self.osd = OSDWindow()
        self.settings_file = "settings.json"
        
        threading.Thread(target=run_flask, daemon=True).start()
        self.apply_theme()
        self.init_ui()
        self.load_settings()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI'; font-size: 14px; }
            QTabWidget::pane { border: 1px solid #313244; background: #181825; border-radius: 5px;}
            QTabBar::tab { background: #313244; color: #a6adc8; padding: 10px 20px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #89b4fa; color: #11111b; font-weight: bold; }
            QPushButton { background-color: #89b4fa; color: #11111b; border: none; padding: 8px 12px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:disabled { background-color: #45475a; color: #6c7086; }
            QPushButton#btnEye { background-color: #45475a; color: #cdd6f4; font-size: 16px; border-radius: 4px; }
            QPushButton#btnEye:hover { background-color: #585b70; }
            QPushButton#btnStart { background-color: #a6e3a1; color: #11111b; font-size: 16px; padding: 12px; border-radius: 6px; }
            QPushButton#btnStart:hover { background-color: #94e2d5; }
            QPushButton#btnStop { background-color: #f38ba8; color: #11111b; font-size: 16px; padding: 12px; border-radius: 6px; }
            QPushButton#btnStop:hover { background-color: #eba0ac; }
            QComboBox, QLineEdit { background-color: #313244; border: 1px solid #45475a; padding: 6px; border-radius: 4px; color: #cdd6f4; }
            QLineEdit:disabled, QComboBox:disabled { background-color: #181825; color: #585b70; border: 1px solid #313244; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 15px; font-weight: bold; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #89b4fa; }
            QSlider::groove:horizontal { border: 1px solid #313244; height: 8px; background: #45475a; border-radius: 4px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 18px; margin: -5px 0; border-radius: 9px; }
        """)

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # --- SEKME 1: ÇALIŞMA MOTORLARI ---
        tab_general = QWidget()
        layout_general = QVBoxLayout()
        group_mode = QGroupBox("Mimarî ve Motor Seçimi")
        form_mode = QFormLayout()
        
        self.combo_stt = QComboBox()
        self.combo_stt.addItems(["Offline (Lokal Whisper)", "Online (Google)"]) # Whisper varsayılan
        self.combo_engine = QComboBox()
        self.combo_engine.addItems(["Offline (Lokal Helsinki-NLP)", "Google (Ücretsiz)", "DeepL (API Key)"])
        self.combo_engine.currentTextChanged.connect(self.toggle_api_key)
        
        languages = ["tr", "en", "de", "fr", "es", "it", "ru", "ja"]
        self.combo_src = QComboBox()
        self.combo_src.addItems(languages)
        self.combo_dest = QComboBox()
        self.combo_dest.addItems(languages)
        self.combo_dest.setCurrentText("en")
        
        form_mode.addRow("Ses Tanıma (STT):", self.combo_stt)
        form_mode.addRow("Çeviri Servisi:", self.combo_engine)
        form_mode.addRow("Kaynak Dil:", self.combo_src)
        form_mode.addRow("Hedef Dil:", self.combo_dest)
        group_mode.setLayout(form_mode)
        layout_general.addWidget(group_mode)

        group_mic = QGroupBox("Mikrofon Hassasiyeti")
        form_mic = QFormLayout()
        self.slider_pause = QSlider(Qt.Horizontal)
        self.slider_pause.setRange(4, 20)
        self.slider_pause.setValue(8)
        self.lbl_pause_val = QLabel("0.8 sn")
        self.slider_pause.valueChanged.connect(lambda v: self.lbl_pause_val.setText(f"{v/10.0} sn"))
        
        pause_layout = QHBoxLayout()
        pause_layout.addWidget(self.slider_pause)
        pause_layout.addWidget(self.lbl_pause_val)
        form_mic.addRow("Cümle Bitiş Beklemesi:", pause_layout)
        group_mic.setLayout(form_mic)
        layout_general.addWidget(group_mic)
        
        layout_general.addStretch()
        tab_general.setLayout(layout_general)

        # --- SEKME 2: OSD ---
        tab_osd = QWidget()
        layout_osd = QVBoxLayout()
        group_osd = QGroupBox("Masaüstü Altyazı (OSD)")
        form_osd = QFormLayout()
        self.chk_osd = QCheckBox("OSD Penceresini Aktifleştir")
        self.chk_osd.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        self.chk_osd.stateChanged.connect(self.update_osd_state)
        
        self.combo_screen = QComboBox()
        for idx, screen in enumerate(QApplication.screens()):
            geom = screen.geometry()
            self.combo_screen.addItem(f"Ekran {idx + 1} ({geom.width()}x{geom.height()})")
        
        self.combo_position = QComboBox()
        self.combo_position.addItems(["Alt (Bottom)", "Üst (Top)"])
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(70)
        self.slider_opacity.valueChanged.connect(self.osd.set_opacity)
        
        self.combo_screen.currentIndexChanged.connect(self.update_osd_state)
        self.combo_position.currentIndexChanged.connect(self.update_osd_state)
        
        form_osd.addRow("", self.chk_osd)
        form_osd.addRow("Hedef Ekran:", self.combo_screen)
        form_osd.addRow("Konum:", self.combo_position)
        form_osd.addRow("Şeffaflık:", self.slider_opacity)
        group_osd.setLayout(form_osd)
        layout_osd.addWidget(group_osd)
        layout_osd.addStretch()
        tab_osd.setLayout(layout_osd)

        # --- SEKME 3: SRT KAYIT ---
        tab_record = QWidget()
        layout_record = QVBoxLayout()
        group_record = QGroupBox("DaVinci / YouTube İçin SRT Kaydı")
        form_record = QFormLayout()
        self.chk_srt = QCheckBox("Canlı Konuşmaları Kaydet")
        self.txt_srt_path = QLineEdit(os.path.join(os.path.expanduser("~"), "Desktop", "yayinaltyazi.srt"))
        btn_browse_srt = QPushButton("Gözat")
        btn_browse_srt.clicked.connect(self.browse_srt)
        path_layout_srt = QHBoxLayout()
        path_layout_srt.addWidget(self.txt_srt_path)
        path_layout_srt.addWidget(btn_browse_srt)
        form_record.addRow("", self.chk_srt)
        form_record.addRow("Dosya:", path_layout_srt)
        group_record.setLayout(form_record)
        layout_record.addWidget(group_record)
        layout_record.addStretch()
        tab_record.setLayout(layout_record)

        # --- SEKME 4: API & MODELLER (VE YENİ FİLTRELER) ---
        tab_api = QWidget()
        layout_api = QVBoxLayout()
        
        self.group_api = QGroupBox("DeepL Kimlik Bilgileri")
        form_api = QFormLayout()
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setPlaceholderText("DeepL Key girin...")
        self.txt_api_key.setEchoMode(QLineEdit.Password)
        self.txt_api_key.setEnabled(False) 
        
        self.btn_eye = QPushButton("👁️")
        self.btn_eye.setObjectName("btnEye")
        self.btn_eye.setFixedWidth(40)
        self.btn_eye.clicked.connect(self.toggle_password_visibility)
        
        api_row = QHBoxLayout()
        api_row.addWidget(self.txt_api_key)
        api_row.addWidget(self.btn_eye)
        form_api.addRow("Key:", api_row)
        self.group_api.setLayout(form_api)
        layout_api.addWidget(self.group_api)

        # YENİ EKLENEN KISIM: Halüsinasyon Filtreleri
        self.group_filters = QGroupBox("Yapay Zeka Halüsinasyon ve VAD Filtreleri")
        form_filters = QFormLayout()
        
        # Karaliste (Blacklist)
        self.txt_blacklist = QLineEdit("elderman, altyazı m.k., amara.org, sync, corrected, çeviri:, altyazı:")
        self.txt_blacklist.setToolTip("Görmek istemediğiniz kelimeleri virgülle ayırarak yazın.")
        
        # VAD Sessizlik Eşiği
        self.slider_vad = QSlider(Qt.Horizontal)
        self.slider_vad.setRange(2, 15) # 200ms - 1500ms (x100)
        self.slider_vad.setValue(5) # 500ms
        self.lbl_vad_val = QLabel("500 ms")
        self.slider_vad.valueChanged.connect(lambda v: self.lbl_vad_val.setText(f"{v*100} ms"))
        
        vad_layout = QHBoxLayout()
        vad_layout.addWidget(self.slider_vad)
        vad_layout.addWidget(self.lbl_vad_val)
        
        form_filters.addRow("Zehirli Kelimeler (Kara Liste):", self.txt_blacklist)
        form_filters.addRow("VAD Sessizlik Engeli (Dip Ses):", vad_layout)
        
        info_filter = QLabel("<i>VAD süresini dip gürültünüze göre artırın. Zehirli kelime listesi Whisper'ın<br>uydurduğu metinleri ekrana yansıtmadan çöpe atar.</i>")
        info_filter.setStyleSheet("color: #a6adc8; font-size: 11px;")
        form_filters.addRow("", info_filter)
        
        self.group_filters.setLayout(form_filters)
        layout_api.addWidget(self.group_filters)

        self.group_whisper = QGroupBox("Lokal STT (Whisper) Boyutu")
        form_whisper = QFormLayout()
        self.combo_whisper_size = QComboBox()
        self.combo_whisper_size.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        self.combo_whisper_size.setCurrentText("large-v3") 
        form_whisper.addRow("Model Boyutu:", self.combo_whisper_size)
        self.group_whisper.setLayout(form_whisper)
        layout_api.addWidget(self.group_whisper)
        
        layout_api.addStretch()
        tab_api.setLayout(layout_api)

        self.tabs.addTab(tab_general, "Genel")
        self.tabs.addTab(tab_osd, "Görünüm")
        self.tabs.addTab(tab_record, "Kayıt (SRT)")
        self.tabs.addTab(tab_api, "API & Filtreler")
        main_layout.addWidget(self.tabs)

        self.lbl_status = QLabel("Durum: Bekliyor (Web: localhost:3333)")
        self.lbl_status.setStyleSheet("color: #a6adc8; font-style: italic; margin-top: 10px;")
        self.btn_start = QPushButton("SİSTEMİ BAŞLAT")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.toggle_translation)

        main_layout.addWidget(self.lbl_status)
        main_layout.addWidget(self.btn_start)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # --- MANTIK ---
    def toggle_password_visibility(self):
        if self.txt_api_key.echoMode() == QLineEdit.Password:
            self.txt_api_key.setEchoMode(QLineEdit.Normal)
            self.btn_eye.setStyleSheet("background-color: #f38ba8; color: #11111b;")
        else:
            self.txt_api_key.setEchoMode(QLineEdit.Password)
            self.btn_eye.setStyleSheet("")

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "stt_mode" in data: self.combo_stt.setCurrentText(data["stt_mode"])
                    if "engine" in data: self.combo_engine.setCurrentText(data["engine"])
                    if "src_lang" in data: self.combo_src.setCurrentText(data["src_lang"])
                    if "dest_lang" in data: self.combo_dest.setCurrentText(data["dest_lang"])
                    if "api_key" in data: self.txt_api_key.setText(data["api_key"])
                    if "whisper_size" in data: self.combo_whisper_size.setCurrentText(data["whisper_size"])
                    if "srt_path" in data: self.txt_srt_path.setText(data["srt_path"])
                    if "pause_thresh" in data: self.slider_pause.setValue(data["pause_thresh"])
                    if "opacity" in data: self.slider_opacity.setValue(data["opacity"])
                    if "osd_enabled" in data: self.chk_osd.setChecked(data["osd_enabled"])
                    if "srt_enabled" in data: self.chk_srt.setChecked(data["srt_enabled"])
                    if "blacklist" in data: self.txt_blacklist.setText(data["blacklist"])
                    if "vad_ms" in data: self.slider_vad.setValue(data["vad_ms"])
            except Exception:
                pass
        self.toggle_api_key(self.combo_engine.currentText())

    def save_settings(self):
        data = {
            "stt_mode": self.combo_stt.currentText(),
            "engine": self.combo_engine.currentText(),
            "api_key": self.txt_api_key.text(), 
            "src_lang": self.combo_src.currentText(),
            "dest_lang": self.combo_dest.currentText(),
            "whisper_size": self.combo_whisper_size.currentText(),
            "srt_path": self.txt_srt_path.text(),
            "pause_thresh": self.slider_pause.value(),
            "opacity": self.slider_opacity.value(),
            "osd_enabled": self.chk_osd.isChecked(),
            "srt_enabled": self.chk_srt.isChecked(),
            "blacklist": self.txt_blacklist.text(),
            "vad_ms": self.slider_vad.value()
        }
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    def closeEvent(self, event):
        self.save_settings() 
        if self.worker:
            self.worker.stop()
        event.accept()

    def toggle_api_key(self, text):
        is_deepl = "API Key" in text
        self.txt_api_key.setEnabled(is_deepl)
        self.btn_eye.setEnabled(is_deepl)
        if not is_deepl:
            self.txt_api_key.setEchoMode(QLineEdit.Password)
            self.btn_eye.setStyleSheet("")
            
    def browse_srt(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Altyazı Dosyasını Kaydet", self.txt_srt_path.text(), "SubRip Subtitle (*.srt)")
        if file_path: self.txt_srt_path.setText(file_path)

    def update_osd_state(self):
        if self.chk_osd.isChecked():
            self.osd.apply_positioning(self.combo_screen.currentIndex(), self.combo_position.currentText())
            self.osd.show()
        else:
            self.osd.hide()

    def toggle_translation(self):
        self.save_settings() 
        if self.worker is None or not self.worker.running:
            self.btn_start.setText("BAŞLATILIYOR...")
            self.btn_start.setEnabled(False)
            QApplication.processEvents() 
            
            self.worker = TranslationWorker(
                stt_mode=self.combo_stt.currentText(), 
                whisper_size=self.combo_whisper_size.currentText(),
                trans_engine=self.combo_engine.currentText(),
                api_key=self.txt_api_key.text(),
                src_lang=self.combo_src.currentText(), 
                dest_lang=self.combo_dest.currentText(), 
                pause_thresh=(self.slider_pause.value() / 10.0),
                srt_enabled=self.chk_srt.isChecked(),
                srt_path=self.txt_srt_path.text(),
                blacklist_str=self.txt_blacklist.text(),
                vad_ms=(self.slider_vad.value() * 100) # Örneğin slider 5 ise 500 ms yapar
            )
            self.worker.update_text.connect(self.osd.update_text)
            self.worker.status_update.connect(self.lbl_status.setText)
            self.worker.start()
            
            self.btn_start.setEnabled(True)
            self.btn_start.setText("SİSTEMİ DURDUR")
            self.btn_start.setObjectName("btnStop")
            self.setStyleSheet(self.styleSheet())
            for i in range(self.tabs.count()):
                if i != 1: self.tabs.setTabEnabled(i, False)
        else:
            self.btn_start.setText("KAPATILIYOR (BEKLEYİN)...")
            self.btn_start.setEnabled(False)
            QApplication.processEvents()
            
            self.worker.stop()
            self.worker = None
            
            self.btn_start.setEnabled(True)
            self.btn_start.setText("SİSTEMİ BAŞLAT")
            self.btn_start.setObjectName("btnStart")
            self.setStyleSheet(self.styleSheet())
            self.lbl_status.setText("Durum: Durduruldu ve Bellek Temizlendi.")
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec_())