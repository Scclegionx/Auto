package com.example.Auto_BE.service;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.messaging.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

@Service
public class FcmService {
    public static void initialize() {
        try {
            // Kiểm tra xem Firebase App đã được khởi tạo chưa
            if (FirebaseApp.getApps().isEmpty()) {
                // Đọc service account key từ file path (ưu tiên Docker mount)
                InputStream serviceAccount;
                try {
                    // Thử đọc từ Docker mount path trước
                    serviceAccount = new java.io.FileInputStream("/app/firebase-service-account.json");
                    System.out.println("Loading Firebase config from Docker mount: /app/firebase-service-account.json");
                } catch (java.io.FileNotFoundException e) {
                    // Fallback to resources folder
                    serviceAccount = new ClassPathResource("firebase-service-account.json").getInputStream();
                    System.out.println("Loading Firebase config from resources folder");
                }

                FirebaseOptions options = FirebaseOptions.builder()
                        .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                        .build();

                FirebaseApp.initializeApp(options);
                System.out.println("Firebase Admin SDK initialized successfully");
            }
        } catch (IOException e) {
            System.err.println("Error initializing Firebase: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public boolean sendNotification(String deviceToken, String title, String body) {
        try {
            Message message = Message.builder()
                    .setToken(deviceToken)
                    .setNotification(Notification.builder()
                            .setTitle(title)
                            .setBody(body)
                            .build())
                    .build();

            String response = FirebaseMessaging.getInstance().send(message);
            System.out.println("Sent message: " + response);
            return true;
        } catch (FirebaseMessagingException e) {
            System.err.println("Error sending FCM message: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    // NEW: gửi nhiều token cùng lúc (tối đa 500)
    public BatchResponse sendNotification(List<String> deviceTokens, String title, String body) throws FirebaseMessagingException {
        MulticastMessage message = MulticastMessage.builder()
                .addAllTokens(deviceTokens)
                .setNotification(Notification.builder().setTitle(title).setBody(body).build())
                .build();
        return FirebaseMessaging.getInstance().sendEachForMulticast(message);
    }
}