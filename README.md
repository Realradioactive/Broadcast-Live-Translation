# 🌐 Canlı Çeviri ve Yayıncı Altyazı Merkezi (Live Subtitle & Translation Center)

> Yerel GPU hızlandırmalı, sıfır gecikmeli, çevrimdışı (offline) veya çevrimiçi (online) çalışabilen, masaüstü şeffaf altyazı (OSD) ve SRT kayıt özellikli profesyonel yayıncı aracı.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green?style=for-the-badge&logo=qt&logoColor=white)
![Whisper](https://img.shields.io/badge/AI-Faster_Whisper-orange?style=for-the-badge&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)


---

## 🚀 Öne Çıkan Özellikler

* **Çift Mod Mimarisi (Online & Offline):** İster Google STT ve Google/DeepL çeviri servislerini kullanın, ister internet kablosunu çekip **OpenAI Faster-Whisper** ve **Helsinki-NLP** modelleriyle tamamen lokalde (GPU destekli) çalışın.
* **Asenkron Kuyruk (Queue) Mimarisi:** Dinleme (Producer) ve Çeviri/İşleme (Consumer) süreçleri tamamen izole edilmiştir. İnternet veya model gecikmelerinde asla kelime/cümle kaybı yaşanmaz.
* **Masaüstü Şeffaf Altyazı (OSD):** Oyun oynarken (örn. *Rust*, *CS:GO*) veya yayın yaparken, istediğiniz ekranda üstte kalan (Always-on-Top), şeffaflığı ayarlanabilir modern altyazı penceresi.
* **DaVinci Resolve & YouTube Uyumlu SRT Kaydı:** Konuşmalarınızı milisaniyelik zaman damgalarıyla anlık olarak `.srt` formatına disk yazar. Oyun seanslarınız bittiğinde altyazınız montaja hazırdır.
* **Gelişmiş Halüsinasyon ve VAD Filtreleri:** Yapay zekanın sessizlikte uydurabileceği ("Amara.org", "elderman" vb.) korsan altyazı imzalarını **Kara Liste (Blacklist)** filtresiyle otomatik olarak çöpe atar.
* **Web Sunucu Desteği:** Yerel ağ üzerinden (localhost:3333) altyazıları başka bir tarayıcıya veya OBS tarayıcı kaynağına (Browser Source) aktarmanıza olanak tanır.
* **Güvenli Arayüz (Secure UI):** DeepL API anahtarlarınızı şifreli (Password Mode) tutar ve göz butonuyla gizliliğini korur. Tüm ayarlarınızı `settings.json` ile hatırlar.

---
<img width="673" height="677" alt="image" src="https://github.com/user-attachments/assets/95a3ca61-895e-4fa4-9c77-94cc4dc022fc" />
## 🛠️ Sistem Gereksinimleri ve Teknolojiler

* **Python:** 3.11+
* **Arayüz:** PyQt5
* **Ses Tanıma (STT):** `faster-whisper` (CUDA/GPU Destekli) veya Google Speech Recognition
* **Çeviri Motorları:** `deep-transformers` (Helsinki-NLP MarianMT), Google Translate, DeepL API
* **Web Arayüzü:** Flask

---

## 📦 Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları takip edin:

1. **Repoyu klonlayın veya indirin:**
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/yayin-cevirmeni.git](https://github.com/KULLANICI_ADINIZ/yayin-cevirmeni.git)
   cd yayin-cevirmeni
İzole bir sanal ortam (venv) oluşturun ve aktif edin:

Bash
python -m venv venv
# Windows için:
venv\Scripts\activate
Gerekli kütüphaneleri yükleyin:

Bash
pip install PyQt5 Flask SpeechRecognition deep-translator transformers torch faster-whisper numpy pyaudio requests
Uygulamayı başlatın:

Bash
python app.py
⚙️ .Exe (Çalıştırılabilir Dosya) Derleme
Python ve sanal ortam bağımlılığı olmadan, çift tıklamayla masaüstünde çalıştırmak için PyInstaller kullanabilirsiniz:

Proje klasörüne .ico uzantılı bir ikon ekleyin (örn: icon.ico).

Terminalde şu komutu çalıştırın:

Bash
pyinstaller --noconfirm --onedir --windowed --icon "icon.ico" --name "YayinCevirmeni" app.py
dist/YayinCevirmeni klasörünün içindeki YayinCevirmeni.exe dosyasını dilediğiniz gibi kullanabilirsiniz.

🎯 Kullanım İpuçları
GPU Hızlandırma: Bilgisayarınızda NVIDIA ekran kartı varsa, Faster-Whisper otomatik olarak CUDA çekirdeklerini kullanır ve çeviriler milisaniyeler içinde ekrana gelir.

Model Seçimi: Canlı yayın performansı için medium veya large-v3 modellerini tercih edebilirsiniz.

📄 Lisans
Bu proje MIT Lisansı altında açık kaynak olarak sunulmuştur. Dilediğiniz gibi geliştirebilir, değiştirebilir ve ticari/bireysel olarak kullanabilirsiniz.

