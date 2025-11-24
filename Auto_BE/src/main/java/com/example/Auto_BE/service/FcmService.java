package com.example.Auto_BE.service;

import com.example.Auto_BE.entity.DeviceToken;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.repository.DeviceTokenRepository;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.messaging.*;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class FcmService {
    
    @Autowired
    private DeviceTokenRepository deviceTokenRepository;
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
    
    /**
     * Gửi notification tin nhắn chat mới đến tất cả devices của user
     */
    public void sendChatNotification(
            User receiver,
            String senderName,
            String messageContent,
            Long chatId
    ) {
        // Lấy tất cả FCM tokens của receiver
        List<DeviceToken> deviceTokens = deviceTokenRepository.findByUserAndIsActive(receiver, true);
        
        if (deviceTokens.isEmpty()) {
            System.out.println("⚠️ No active device tokens found for user: " + receiver.getEmail());
            return;
        }
        
        System.out.println("📱 Sending FCM notification to " + deviceTokens.size() + " devices of " + receiver.getEmail());
        
        // Build notification data
        Map<String, String> data = new HashMap<>();
        data.put("type", "chat_message");
        data.put("chatId", String.valueOf(chatId));
        data.put("senderName", senderName);
        data.put("messageContent", messageContent);
        
        // Gửi đến từng device
        for (DeviceToken deviceToken : deviceTokens) {
            try {
                String token = deviceToken.getFcmToken();
                
                if (token == null || token.isEmpty()) {
                    continue;
                }
                
                // Build message
                Message message = Message.builder()
                        .setToken(token)
                        .setNotification(Notification.builder()
                                .setTitle(senderName)
                                .setBody(messageContent)
                                .build())
                        .putAllData(data)
                        .setAndroidConfig(AndroidConfig.builder()
                                .setPriority(AndroidConfig.Priority.HIGH)
                                .setNotification(AndroidNotification.builder()
                                        .setSound("default")
                                        .setClickAction("OPEN_CHAT")
                                        .build())
                                .build())
                        .build();
                
                // Send message
                String response = FirebaseMessaging.getInstance().send(message);
                System.out.println("✅ FCM sent successfully to device: " + deviceToken.getDeviceName());
                
            } catch (FirebaseMessagingException e) {
                System.err.println("❌ Failed to send FCM to device " + deviceToken.getDeviceName() + ": " + e.getMessage());
                
                // Nếu token không hợp lệ, vô hiệu hóa nó
                if (e.getErrorCode().equals("invalid-registration-token") || 
                    e.getErrorCode().equals("registration-token-not-registered")) {
                    deviceToken.setIsActive(false);
                    deviceTokenRepository.save(deviceToken);
                    System.out.println("🗑️ Deactivated invalid token for device: " + deviceToken.getDeviceName());
                }
            } catch (Exception e) {
                System.err.println("❌ Unexpected error sending FCM: " + e.getMessage());
            }
        }
    }
}