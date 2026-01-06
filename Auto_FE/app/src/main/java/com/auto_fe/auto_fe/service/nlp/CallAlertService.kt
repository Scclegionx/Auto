package com.auto_fe.auto_fe.automation.phone

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.IBinder
import android.provider.ContactsContract
import android.util.Log
import androidx.core.app.NotificationCompat
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.audio.TTSManager
import kotlinx.coroutines.*

class CallAlertService : Service() {

    companion object {
        private const val TAG = "CallAlertService"
        private const val CHANNEL_ID = "call_alert_channel"
        private const val NOTIF_ID = 2024

        const val ACTION_INCOMING_CALL = "ACTION_INCOMING_CALL"
        const val ACTION_CALL_ENDED = "ACTION_CALL_ENDED"
        const val EXTRA_PHONE_NUMBER = "EXTRA_PHONE_NUMBER"
    }

    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private var audioManager: AudioManager? = null

    // Lưu lại trạng thái volume cũ để khôi phục
    private var previousRingVolume: Int? = null
    private var previousMusicVolume: Int? = null

    // Flag để tránh báo nhiều lần cho cùng 1 cuộc gọi
    private var isAlerting = false

    // Audio Focus Request (cho Android O trở lên)
    private var focusRequest: AudioFocusRequest? = null

    override fun onCreate() {
        super.onCreate()
        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Start Foreground ngay lập tức
        startForeground(NOTIF_ID, buildNotification("Đang giám sát cuộc gọi..."))

        when (intent?.action) {
            ACTION_INCOMING_CALL -> {
                val number = intent.getStringExtra(EXTRA_PHONE_NUMBER)
                handleIncomingCall(number)
            }
            ACTION_CALL_ENDED -> {
                restoreAudioSettings()
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun handleIncomingCall(number: String?) {
        if (isAlerting) return
        isAlerting = true

        scope.launch {
            // 1. Check Contact (Background Thread)
            val isUnknown = isUnknownNumber(number)

            if (isUnknown) {
                withContext(Dispatchers.Main) {
                    // Cập nhật thông báo
                    val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                    nm.notify(NOTIF_ID, buildNotification("Cảnh báo: Số lạ gọi đến!"))

                    // 2. Chỉnh Volume & Audio Focus
                    if (prepareAudioForAlert()) {
                        // 3. Nói cảnh báo bằng TTSManager
                        speakWarning()
                    }
                }
            } else {
                Log.d(TAG, "Known contact. Service stopping.")
                stopSelf()
            }
        }
    }

    /**
     * Hạ volume chuông, tăng volume giọng nói, xin Audio Focus
     * @return true nếu thành công
     */
    private fun prepareAudioForAlert(): Boolean {
        return try {
            audioManager?.let { am ->
                // Lưu volume cũ
                previousRingVolume = am.getStreamVolume(AudioManager.STREAM_RING)
                previousMusicVolume = am.getStreamVolume(AudioManager.STREAM_MUSIC)

                // Hạ chuông (30%)
                val maxRing = am.getStreamMaxVolume(AudioManager.STREAM_RING)
                val targetRing = (maxRing * 0.3f).toInt().coerceAtLeast(1)
                am.setStreamVolume(AudioManager.STREAM_RING, targetRing, 0)

                // Tăng media (80%) để TTS nói to
                val maxMusic = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
                val targetMusic = (maxMusic * 0.8f).toInt()
                am.setStreamVolume(AudioManager.STREAM_MUSIC, targetMusic, 0)

                // Xin Audio Focus (Ducking - dìm âm thanh khác xuống)
                requestAudioFocus()
                true
            } ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Error adjusting volume: ${e.message}")
            false
        }
    }

    private fun speakWarning() {
        val ttsManager = TTSManager.getInstance(this)

        // Đọc 2 lần cho chắc
        ttsManager.speak("Cảnh báo. Có số lạ gọi đến.")

        // Tự động tắt sau 6 giây
        android.os.Handler(mainLooper).postDelayed({
            restoreAudioSettings()
            stopSelf()
        }, 6000)
    }

    private fun restoreAudioSettings() {
        try {
            audioManager?.let { am ->
                previousRingVolume?.let {
                    am.setStreamVolume(AudioManager.STREAM_RING, it, 0)
                }
                previousMusicVolume?.let {
                    am.setStreamVolume(AudioManager.STREAM_MUSIC, it, 0)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error restoring volume: ${e.message}")
        }

        abandonAudioFocus()
        isAlerting = false
    }

    // --- Check danh bạ (Background) ---
    private suspend fun isUnknownNumber(number: String?): Boolean = withContext(Dispatchers.IO) {
        if (number.isNullOrBlank()) return@withContext true

        var isUnknown = true
        val uri = ContactsContract.PhoneLookup.CONTENT_FILTER_URI.buildUpon().appendPath(number).build()
        val projection = arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME)

        try {
            val cursor: Cursor? = contentResolver.query(uri, projection, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    isUnknown = false // Tìm thấy tên -> Người quen
                    Log.d(TAG, "Contact found: ${it.getString(0)}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Lookup failed", e)
        }
        return@withContext isUnknown
    }

    // --- Audio Focus Utils ---
    private fun requestAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val attrs = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                .setAudioAttributes(attrs)
                .setOnAudioFocusChangeListener { /* No-op */ }
                .build()
            audioManager?.requestAudioFocus(focusRequest!!)
        } else {
            @Suppress("DEPRECATION")
            audioManager?.requestAudioFocus(
                null,
                AudioManager.STREAM_MUSIC,
                AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK
            )
        }
    }

    private fun abandonAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            focusRequest?.let { audioManager?.abandonAudioFocusRequest(it) }
        } else {
            @Suppress("DEPRECATION")
            audioManager?.abandonAudioFocus(null)
        }
    }

    private fun buildNotification(text: String): Notification {
        // Đổi icon thành R.drawable.ic_mic_white hoặc icon app của bạn
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_mic_white)
            .setContentTitle("Trợ lý cuộc gọi")
            .setContentText(text)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Call Alerts", NotificationManager.IMPORTANCE_LOW)
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        restoreAudioSettings()
        scope.cancel()
        super.onDestroy()
    }
}