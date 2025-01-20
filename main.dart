import 'dart:convert';
import 'dart:html'; // Flutter Web에서 파일 업로드에 필요

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ImageUploadScreen(),
    );
  }
}

class ImageUploadScreen extends StatefulWidget {
  @override
  _ImageUploadScreenState createState() => _ImageUploadScreenState();
}

class _ImageUploadScreenState extends State<ImageUploadScreen> {
  String? _imageBase64;
  String _response = "";

  // 이미지를 업로드하는 함수
  Future<void> uploadImage() async {
    if (_imageBase64 == null) {
      setState(() {
        _response = "이미지를 선택하세요.";
      });
      return;
    }

    final url = Uri.parse('http://<ip주소입력>:5000/upload');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image': _imageBase64}),
      );

      if (response.statusCode == 200) {
        setState(() {
          _response = "응답: ${response.body}";
        });
      } else {
        setState(() {
          _response = "서버 오류: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _response = "에러 발생: $e";
      });
    }
  }

  // 이미지 선택 (HTML File Input 사용)
  Future<void> pickImage() async {
    FileUploadInputElement uploadInput = FileUploadInputElement();
    uploadInput.accept = 'image/*'; // 이미지 파일만 허용
    uploadInput.click();

    uploadInput.onChange.listen((e) async {
      final files = uploadInput.files;
      if (files != null && files.isNotEmpty) {
        final reader = FileReader();
        reader.readAsDataUrl(files[0]);
        reader.onLoadEnd.listen((_) {
          setState(() {
            _imageBase64 = reader.result.toString().split(',').last; // Base64 인코딩
          });
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Flutter Web 이미지 업로드')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _imageBase64 != null
                ? Text("이미지가 선택되었습니다.")
                : Text("이미지를 선택하세요."),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: pickImage,
              child: Text("이미지 선택"),
            ),
            ElevatedButton(
              onPressed: uploadImage,
              child: Text("서버로 전송"),
            ),
            SizedBox(height: 16),
            Text(_response),
          ],
        ),
      ),
    );
  }
}
