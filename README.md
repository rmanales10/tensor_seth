# 🤖 Tensor Seth — Vision AI

<p align="center">
  <img src="lib/screen.png" width="260" alt="Tensor Seth Screenshot"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter&logoColor=white" />
  <img src="https://img.shields.io/badge/TFLite-0.12.1-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Android-minSdk%2026-3DDC84?logo=android&logoColor=white" />
  <img src="https://img.shields.io/badge/Release%20APK-74.6MB-135BEC" />
</p>

> **Tensor Seth** is a Flutter mobile app that uses an on-device TensorFlow Lite model to identify objects in photos — then plays a matching sound for each detected class.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📷 **Camera & Gallery** | Pick photos from your camera or gallery |
| 🧠 **On-device AI** | Runs TFLite inference fully offline — no internet required |
| 🎯 **4-class detection** | Identifies **Cat**, **Dog**, **Human**, and **Other** |
| 🔊 **Sound feedback** | Plays a unique audio clip for each detected class |
| 📊 **Confidence bar** | Visual indicator showing model confidence (%) |
| 🌑 **Dark UI** | Glassmorphism design with neon blue accents |

---

## 🗂️ Project Structure

```
tensor_seth/
├── lib/
│   └── main.dart               # Full app — UI + TFLite pipeline
├── assets/
│   ├── model.tflite            # TFLite classification model
│   ├── labels.txt              # Class labels (cat, dog, others, human)
│   └── sounds/
│       ├── cat.mp3
│       ├── dog.mp3
│       ├── human.mp3
│       └── other.mp3
├── android/
│   └── app/
│       └── build.gradle.kts    # compileSdk 36, minSdk 26, NDK 27
└── pubspec.yaml
```

---

## 🧠 How the TFLite Pipeline Works

```
User taps Camera or Gallery
         │
         ▼
 image_picker opens picker
         │
         ▼
 Selected image displayed in scanner box
         │
         ▼
 Image decoded → resized to model input shape (e.g. 224×224)
 Normalised to [0.0, 1.0] float32
         │
         ▼
 ┌───────────────────────┐
 │   model.tflite        │  Input:  Float32List [1×H×W×3]
 │   (MobileNet-style)   │  Output: List<double> [1×numClasses]
 └───────────────────────┘
         │
         ▼
 Argmax → highest confidence class
         │
         ▼
 Result card shown (label + confidence bar)
         │
         ▼
 audioplayers plays matching .mp3
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| [`tflite_flutter`](https://pub.dev/packages/tflite_flutter) | `^0.12.1` | TFLite inference engine |
| [`image_picker`](https://pub.dev/packages/image_picker) | `^1.1.2` | Camera / gallery picker |
| [`image`](https://pub.dev/packages/image) | `^4.2.0` | Image decode & resize |
| [`audioplayers`](https://pub.dev/packages/audioplayers) | `^6.1.0` | Sound playback |

---

## 🚀 Getting Started

### Prerequisites

- Flutter SDK `>=3.5.0`
- Android Studio / VS Code
- Android device or emulator (API 26+)

### Clone & Run

```bash
# Clone the repository
git clone <your-repo-url>
cd tensor_seth

# Install dependencies
flutter pub get

# Run in debug mode
flutter run

# Build release APK
flutter build apk --release
# → build/app/outputs/flutter-apk/app-release.apk
```

---

## ⚙️ Android Configuration

| Setting | Value | Reason |
|---|---|---|
| `compileSdk` | `36` | Required by `tflite_flutter 0.12.1` |
| `minSdk` | `26` | Required by `tflite_flutter` |
| `ndkVersion` | `27.0.12077973` | Required by all native plugins |

These are set in [`android/app/build.gradle.kts`](android/app/build.gradle.kts).

---

## 🏷️ Labels

The model classifies images into 4 groups defined in [`assets/labels.txt`](assets/labels.txt):

```
0 cat
1 dog
2 others
3 human
```

Each label maps to a corresponding sound file in `assets/sounds/`.

---

## 🎨 UI Design

The UI is ported from a custom HTML/TailwindCSS design (`lib/code.html`) into Flutter widgets:

| HTML Concept | Flutter Widget |
|---|---|
| Dark gradient background | `LinearGradient` in `Container` |
| Glassmorphism panels | `BackdropFilter` + `ClipRRect` |
| Neon glow border | `BoxShadow` on `BoxDecoration` |
| Scanner corner brackets | Custom `CustomPainter` |
| Press-to-scale animation | `ScaleTransition` + `AnimationController` |
| Ambient glow blobs | Blurred circular `Container` |

**Colour palette:**
- Primary: `#135BEC` (electric blue)
- Background: `#101622 → #161D2B → #0A0F18`

---

## 📱 Permissions

Add to `AndroidManifest.xml` (already configured):

```xml
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
<uses-permission android:name="android.permission.CAMERA" />
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|---|---|
| `UnmodifiableUint8ListView` compile error | Use `tflite_flutter ^0.12.1` (not `0.10.4`) |
| `minSdkVersion too low` | Set `minSdk = 26` in `build.gradle.kts` |
| NDK version mismatch | Set `ndkVersion = "27.0.12077973"` |
| `Image picker already active` | Handled — tap guard prevents double-launch |
| Model load fails | Ensure `assets/model.tflite` is listed in `pubspec.yaml` |

---

## 📄 License

This project is for educational and demonstration purposes.

---

<p align="center">Built with ❤️ using Flutter + TensorFlow Lite</p>
