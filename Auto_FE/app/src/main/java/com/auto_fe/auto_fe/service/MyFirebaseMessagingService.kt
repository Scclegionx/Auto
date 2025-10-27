package com.auto_fe.auto_fe.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.auto_fe.auto_fe.MainActivity
import com.auto_fe.auto_fe.R
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class MyFirebaseMessagingService : FirebaseMessagingService() {

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.d(TAG, "New FCM Token: $token")
        
        // Lưu token vào SharedPreferences
        saveFCMToken(token)
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        super.onMessageReceived(remoteMessage)
        
        Log.d(TAG, "========== FCM Message Received ==========")
        Log.d(TAG, "Message received from: ${remoteMessage.from}")
        
        // Xử lý notification payload
        remoteMessage.notification?.let {
            Log.d(TAG, "Notification Title: ${it.title}")
            Log.d(TAG, "Notification Body: ${it.body}")
            sendNotification(it.title ?: "Thông báo", it.body ?: "")
        }
        
        // Xử lý data payload (khi không có notification payload)
        if (remoteMessage.data.isNotEmpty()) {
            Log.d(TAG, "Message data payload: ${remoteMessage.data}")
            
            // Nếu không có notification payload, tạo notification từ data
            if (remoteMessage.notification == null) {
                val title = remoteMessage.data["title"] ?: "💊 Nhắc nhở uống thuốc"
                val body = remoteMessage.data["body"] ?: remoteMessage.data["message"] ?: ""
                Log.d(TAG, "Creating notification from data payload")
                sendNotification(title, body)
            }
            
            handleDataPayload(remoteMessage.data)
        }
        
        Log.d(TAG, "==========================================")
    }

    private fun sendNotification(title: String, messageBody: String) {
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this, 
            0, 
            intent,
            PendingIntent.FLAG_ONE_SHOT or PendingIntent.FLAG_IMMUTABLE
        )

        val channelId = getString(R.string.default_notification_channel_id)
        
        // Tạo unique notification ID để mỗi notification hiển thị riêng
        val notificationId = System.currentTimeMillis().toInt()
        
        val notificationBuilder = NotificationCompat.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(title)
            .setContentText(messageBody)
            .setStyle(NotificationCompat.BigTextStyle().bigText(messageBody)) // Hiển thị full text
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setDefaults(NotificationCompat.DEFAULT_ALL) // Sound + Vibration + Lights
            .setContentIntent(pendingIntent)

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        // Tạo notification channel cho Android O+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Medication Reminders",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Notifications for medication reminders"
                enableVibration(true)
                enableLights(true)
                setShowBadge(true)
            }
            notificationManager.createNotificationChannel(channel)
        }

        Log.d(TAG, "Sending notification with ID: $notificationId, Title: $title")
        notificationManager.notify(notificationId, notificationBuilder.build())
    }

    private fun handleDataPayload(data: Map<String, String>) {
        // Xử lý custom data từ notification
        val prescriptionId = data["prescriptionId"]
        val medicineId = data["medicineId"]
        
        Log.d(TAG, "Prescription ID: $prescriptionId")
        Log.d(TAG, "Medicine ID: $medicineId")
        
        // TODO: Navigate to specific prescription/medicine detail
    }

    private fun saveFCMToken(token: String) {
        val sharedPref = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        with(sharedPref.edit()) {
            putString("fcm_token", token)
            apply()
        }
        Log.d(TAG, "FCM Token saved to SharedPreferences")
    }

    companion object {
        private const val TAG = "MyFCMService"
        
        fun getSavedFCMToken(context: Context): String? {
            val sharedPref = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            return sharedPref.getString("fcm_token", null)
        }
    }
}
