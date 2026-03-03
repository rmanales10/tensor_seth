import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// ─── Entry point ──────────────────────────────────────────────────────────────
void main() {
  runApp(const VisionAIApp());
}

// ─── Colour tokens ────────────────────────────────────────────────────────────
const Color kPrimary = Color(0xFF135BEC);
const Color kBgDark = Color(0xFF101622);
const Color kBgMid = Color(0xFF161D2B);
const Color kBgDeep = Color(0xFF0A0F18);
const Color kSlate400 = Color(0xFF94A3B8);
const Color kSlate300 = Color(0xFFCBD5E1);

// ─── Label → sound asset path ─────────────────────────────────────────────────
const Map<String, String> kSoundMap = {
  'cat': 'sounds/cat.mp3',
  'dog': 'sounds/dog.mp3',
  'human': 'sounds/human.mp3',
  'others': 'sounds/other.mp3',
};

// ─── App root ─────────────────────────────────────────────────────────────────
class VisionAIApp extends StatelessWidget {
  const VisionAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tensor Seth',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(scaffoldBackgroundColor: kBgDark),
      home: const VisionAIHome(),
    );
  }
}

// ─── Detection result ─────────────────────────────────────────────────────────
class DetectionResult {
  final String label;
  final double confidence;
  const DetectionResult(this.label, this.confidence);
}

// ─── Home page (StatefulWidget) ───────────────────────────────────────────────
class VisionAIHome extends StatefulWidget {
  const VisionAIHome({super.key});
  @override
  State<VisionAIHome> createState() => _VisionAIHomeState();
}

class _VisionAIHomeState extends State<VisionAIHome> {
  // ── TFLite ──────────────────────────────────────────────────────────────────
  Interpreter? _interpreter;
  List<String> _labels = [];

  // ── UI state ────────────────────────────────────────────────────────────────
  File? _pickedImage;
  DetectionResult? _result;
  bool _isLoading = false;
  bool _isPicking = false; // guard against double-tap
  String? _errorMsg;

  // ── Audio ───────────────────────────────────────────────────────────────────
  final AudioPlayer _audioPlayer = AudioPlayer();

  // ────────────────────────────────────────────────────────────────────────────
  //  Lifecycle
  // ────────────────────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  @override
  void dispose() {
    _interpreter?.close();
    _audioPlayer.dispose();
    super.dispose();
  }

  // ────────────────────────────────────────────────────────────────────────────
  //  Step 1 — Load model + labels
  // ────────────────────────────────────────────────────────────────────────────
  Future<void> _loadModelAndLabels() async {
    try {
      // Load TFLite model from assets
      final interpreter = await Interpreter.fromAsset('assets/model.tflite');

      // Load labels.txt  (format: "0 cat\n1 dog\n...")
      final raw = await rootBundle.loadString('assets/labels.txt');
      final labels = raw.split('\n').where((l) => l.trim().isNotEmpty).map((l) {
        final parts = l.trim().split(RegExp(r'\s+'));
        return parts.length >= 2 ? parts.sublist(1).join(' ') : parts[0];
      }).toList();

      if (mounted) {
        setState(() {
          _interpreter = interpreter;
          _labels = labels;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _errorMsg = 'Model load failed: $e');
    }
  }

  // ────────────────────────────────────────────────────────────────────────────
  //  Step 2 — Pick image from camera or gallery
  // ────────────────────────────────────────────────────────────────────────────
  Future<void> _pickImage(ImageSource source) async {
    if (_isPicking) return; // prevent double-launch
    setState(() => _isPicking = true);
    try {
      final XFile? xFile = await ImagePicker().pickImage(
        source: source,
        imageQuality: 90,
      );
      if (xFile == null) return;

      setState(() {
        _pickedImage = File(xFile.path);
        _result = null;
        _errorMsg = null;
      });

      await _runInference(_pickedImage!);
    } finally {
      if (mounted) setState(() => _isPicking = false);
    }
  }

  // ────────────────────────────────────────────────────────────────────────────
  //  Step 3 — Run TFLite inference
  // ────────────────────────────────────────────────────────────────────────────
  Future<void> _runInference(File imageFile) async {
    if (_interpreter == null) {
      setState(() => _errorMsg = 'Model not loaded yet, please wait…');
      return;
    }

    setState(() => _isLoading = true);

    try {
      // Read model input shape → [1, H, W, 3]
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final int modelH = inputShape[1];
      final int modelW = inputShape[2];

      // Decode image → resize to model dimensions
      final rawBytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(rawBytes);
      if (decoded == null) throw Exception('Could not decode image');
      final resized = img.copyResize(decoded, width: modelW, height: modelH);

      // Build float32 input tensor  [1, H, W, 3]  normalised to [0, 1]
      // Stored as flat Float32List — tflite_flutter 0.12 accepts it directly
      final inputTensor = Float32List(1 * modelH * modelW * 3);
      int idx = 0;
      for (int y = 0; y < modelH; y++) {
        for (int x = 0; x < modelW; x++) {
          final pixel = resized.getPixel(x, y);
          inputTensor[idx++] = pixel.r / 255.0;
          inputTensor[idx++] = pixel.g / 255.0;
          inputTensor[idx++] = pixel.b / 255.0;
        }
      }

      // Prepare output buffer as List<List<double>> → shape [1, numClasses]
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final numClasses = outputShape[outputShape.length - 1];
      // Build a proper nested list so tflite_flutter can write into it
      final outputBuffer = <List<double>>[List<double>.filled(numClasses, 0.0)];

      // ── Run inference ──────────────────────────────────────────────────────
      _interpreter!.run(inputTensor.buffer, outputBuffer);

      // Find highest confidence class (argmax)
      final scores = outputBuffer[0];
      int maxIdx = 0;
      double maxVal = scores[0];
      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > maxVal) {
          maxVal = scores[i];
          maxIdx = i;
        }
      }

      final label = maxIdx < _labels.length ? _labels[maxIdx] : 'Unknown';
      final result = DetectionResult(label, maxVal);

      setState(() {
        _result = result;
        _isLoading = false;
      });

      // Step 4 — Play the matching sound
      await _playSound(label);
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMsg = 'Inference failed: $e';
      });
    }
  }

  // ────────────────────────────────────────────────────────────────────────────
  //  Step 4 — Play sound matching the detected label
  // ────────────────────────────────────────────────────────────────────────────
  Future<void> _playSound(String label) async {
    final key = label.toLowerCase().replaceAll('.', '').trim();
    final soundPath = kSoundMap[key] ?? kSoundMap['others']!;
    try {
      await _audioPlayer.stop();
      await _audioPlayer.play(AssetSource(soundPath));
    } catch (_) {
      // sound failure is non‑fatal
    }
  }

  // ────────────────────────────────────────────────────────────────────────────
  //  Build
  // ────────────────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          // Gradient background
          _buildGradientBg(),

          // Ambient glow blobs
          Positioned(
            top: -80,
            left: -80,
            child: _GlowBlob(size: 280, color: kPrimary.withAlpha(25)),
          ),
          Positioned(
            bottom: -60,
            right: -60,
            child: _GlowBlob(size: 320, color: kPrimary.withAlpha(13)),
          ),

          // Main content
          SafeArea(
            child: Column(
              children: [
                _AppHeader(modelReady: _interpreter != null),
                Expanded(child: _buildBody()),
                _buildActionButtons(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGradientBg() => Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [kBgDark, kBgMid, kBgDeep],
          ),
        ),
      );

  Widget _buildBody() {
    return SingleChildScrollView(
      child: Column(
        children: [
          const SizedBox(height: 20),
          _ScannerBox(pickedImage: _pickedImage, isLoading: _isLoading),
          const SizedBox(height: 28),
          if (_result != null)
            _ResultCard(result: _result!)
          else if (_errorMsg != null)
            _ErrorBanner(message: _errorMsg!)
          else
            _IdleSubtitle(modelReady: _interpreter != null),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 28),
      child: Row(
        children: [
          Expanded(
            child: _ActionButton(
              icon: Icons.photo_camera,
              label: 'Camera',
              isPrimary: true,
              onTap: () => _pickImage(ImageSource.camera),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: _ActionButton(
              icon: Icons.folder_open,
              label: 'Gallery',
              isPrimary: false,
              onTap: () => _pickImage(ImageSource.gallery),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Header ───────────────────────────────────────────────────────────────────
class _AppHeader extends StatelessWidget {
  final bool modelReady;
  const _AppHeader({required this.modelReady});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            'Tensor Seth',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              letterSpacing: -0.3,
            ),
          ),
          const SizedBox(width: 8),
          // Green = model ready, Amber = still loading
          AnimatedContainer(
            duration: const Duration(milliseconds: 400),
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: modelReady
                  ? const Color(0xFF22C55E)
                  : const Color(0xFFF59E0B),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Scanner / image preview box ──────────────────────────────────────────────
class _ScannerBox extends StatelessWidget {
  final File? pickedImage;
  final bool isLoading;
  const _ScannerBox({this.pickedImage, required this.isLoading});

  @override
  Widget build(BuildContext context) {
    final double size = MediaQuery.of(context).size.width * 0.82;
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white.withAlpha(13),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: kPrimary.withAlpha(128), width: 2),
                  boxShadow: [
                    BoxShadow(
                      color: kPrimary.withAlpha(90),
                      blurRadius: 22,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(22),
                  child: _buildContent(),
                ),
              ),
            ),
          ),
          ..._buildCorners(),
          // Loading overlay
          if (isLoading)
            Positioned.fill(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(140),
                  borderRadius: BorderRadius.circular(24),
                ),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(color: kPrimary),
                      SizedBox(height: 14),
                      Text('Analysing…',
                          style: TextStyle(color: kSlate300, fontSize: 13)),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildContent() {
    if (pickedImage != null) {
      return Image.file(pickedImage!,
          fit: BoxFit.cover, width: double.infinity);
    }
    return Center(
      child: Opacity(
        opacity: 0.6,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: const [
            Icon(Icons.filter_center_focus, color: kPrimary, size: 64),
            SizedBox(height: 14),
            Text(
              'Ready to Scan',
              style: TextStyle(
                color: kSlate300,
                fontSize: 13,
                fontWeight: FontWeight.w500,
                letterSpacing: 1.2,
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildCorners() => [
        Positioned(
            top: -1,
            left: -1,
            child: _Corner(size: 30, thick: 4, top: true, left: true)),
        Positioned(
            top: -1,
            right: -1,
            child: _Corner(size: 30, thick: 4, top: true, left: false)),
        Positioned(
            bottom: -1,
            left: -1,
            child: _Corner(size: 30, thick: 4, top: false, left: true)),
        Positioned(
            bottom: -1,
            right: -1,
            child: _Corner(size: 30, thick: 4, top: false, left: false)),
      ];
}

// ─── Result card ──────────────────────────────────────────────────────────────
class _ResultCard extends StatelessWidget {
  final DetectionResult result;
  const _ResultCard({required this.result});

  IconData _icon() {
    switch (result.label.toLowerCase().replaceAll('.', '')) {
      case 'cat':
        return Icons.pets;
      case 'dog':
        return Icons.cruelty_free;
      case 'human':
        return Icons.person;
      default:
        return Icons.help_outline;
    }
  }

  Color _accent() {
    switch (result.label.toLowerCase().replaceAll('.', '')) {
      case 'cat':
        return const Color(0xFFF59E0B);
      case 'dog':
        return const Color(0xFF22C55E);
      case 'human':
        return const Color(0xFF135BEC);
      default:
        return const Color(0xFF8B5CF6);
    }
  }

  @override
  Widget build(BuildContext context) {
    final accent = _accent();
    final pct = (result.confidence * 100).toStringAsFixed(1);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: accent.withAlpha(30),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: accent.withAlpha(100)),
              boxShadow: [
                BoxShadow(color: accent.withAlpha(50), blurRadius: 20),
              ],
            ),
            child: Row(
              children: [
                // Icon circle
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    color: accent.withAlpha(50),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(_icon(), color: accent, size: 30),
                ),
                const SizedBox(width: 16),
                // Label + confidence bar
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        result.label.toUpperCase(),
                        style: TextStyle(
                          color: accent,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 6),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: result.confidence,
                          minHeight: 6,
                          backgroundColor: Colors.white.withAlpha(25),
                          valueColor: AlwaysStoppedAnimation<Color>(accent),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text('Confidence: $pct%',
                          style:
                              const TextStyle(color: kSlate400, fontSize: 12)),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Icon(Icons.volume_up_rounded,
                    color: accent.withAlpha(180), size: 22),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─── Error banner ─────────────────────────────────────────────────────────────
class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner({required this.message});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.red.withAlpha(30),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.red.withAlpha(100)),
        ),
        child: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent, size: 24),
            const SizedBox(width: 12),
            Expanded(
              child: Text(message,
                  style:
                      const TextStyle(color: Colors.redAccent, fontSize: 13)),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Idle subtitle ────────────────────────────────────────────────────────────
class _IdleSubtitle extends StatelessWidget {
  final bool modelReady;
  const _IdleSubtitle({required this.modelReady});

  @override
  Widget build(BuildContext context) => Column(
        children: [
          Text(
            modelReady ? 'Identify Anything' : 'Loading model…',
            style: const TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              letterSpacing: -0.5,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Pick a photo to detect cat, dog, human or other',
            style: TextStyle(fontSize: 14, color: kSlate400),
            textAlign: TextAlign.center,
          ),
        ],
      );
}

// ─── Action button (Camera / Gallery) ────────────────────────────────────────
class _ActionButton extends StatefulWidget {
  final IconData icon;
  final String label;
  final bool isPrimary;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.isPrimary,
    required this.onTap,
  });

  @override
  State<_ActionButton> createState() => _ActionButtonState();
}

class _ActionButtonState extends State<_ActionButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _scale;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
        vsync: this,
        duration: const Duration(milliseconds: 120),
        lowerBound: 0,
        upperBound: 0.05);
    _scale = Tween<double>(begin: 1.0, end: 0.95)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final content = Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(widget.icon,
            size: 32, color: widget.isPrimary ? Colors.white : kSlate300),
        const SizedBox(height: 8),
        Text(widget.label,
            style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white)),
      ],
    );

    return GestureDetector(
      onTapDown: (_) => _ctrl.forward(),
      onTapUp: (_) {
        _ctrl.reverse();
        widget.onTap();
      },
      onTapCancel: () => _ctrl.reverse(),
      child: ScaleTransition(
        scale: _scale,
        child: widget.isPrimary
            ? Container(
                height: 120,
                decoration: BoxDecoration(
                  color: kPrimary,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: kPrimary.withAlpha(90),
                      blurRadius: 24,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: content,
              )
            : ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: BackdropFilter(
                  filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
                  child: Container(
                    height: 120,
                    decoration: BoxDecoration(
                      color: Colors.white.withAlpha(13),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(color: Colors.white.withAlpha(25)),
                    ),
                    child: content,
                  ),
                ),
              ),
      ),
    );
  }
}

// ─── Corner bracket (custom painter) ─────────────────────────────────────────
class _Corner extends StatelessWidget {
  final double size, thick;
  final bool top, left;
  const _Corner({
    required this.size,
    required this.thick,
    required this.top,
    required this.left,
  });

  @override
  Widget build(BuildContext context) => SizedBox(
        width: size,
        height: size,
        child: CustomPaint(
          painter:
              _CornerPainter(thick: thick, top: top, left: left, radius: 10),
        ),
      );
}

class _CornerPainter extends CustomPainter {
  final double thick, radius;
  final bool top, left;
  _CornerPainter({
    required this.thick,
    required this.top,
    required this.left,
    required this.radius,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = kPrimary
      ..strokeWidth = thick
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final path = Path();
    final w = size.width;
    final h = size.height;
    final r = radius;

    if (top && left) {
      path.moveTo(0, h);
      path.lineTo(0, r);
      path.arcToPoint(Offset(r, 0), radius: Radius.circular(r));
      path.lineTo(w, 0);
    } else if (top && !left) {
      path.moveTo(0, 0);
      path.lineTo(w - r, 0);
      path.arcToPoint(Offset(w, r), radius: Radius.circular(r));
      path.lineTo(w, h);
    } else if (!top && left) {
      path.moveTo(0, 0);
      path.lineTo(0, h - r);
      path.arcToPoint(Offset(r, h), radius: Radius.circular(r));
      path.lineTo(w, h);
    } else {
      path.moveTo(0, h);
      path.lineTo(w - r, h);
      path.arcToPoint(Offset(w, h - r), radius: Radius.circular(r));
      path.lineTo(w, 0);
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}

// ─── Ambient glow blob ────────────────────────────────────────────────────────
class _GlowBlob extends StatelessWidget {
  final double size;
  final Color color;
  const _GlowBlob({required this.size, required this.color});

  @override
  Widget build(BuildContext context) => Container(
        width: size,
        height: size,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
          child: const SizedBox.expand(),
        ),
      );
}
