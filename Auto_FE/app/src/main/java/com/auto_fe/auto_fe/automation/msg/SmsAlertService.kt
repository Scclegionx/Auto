package com.auto_fe.auto_fe.automation.msg

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import android.telephony.TelephonyManager
import androidx.core.app.NotificationCompat
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.audio.AudioRecorder

class SmsAlertService : Service() {

    companion object {
        private const val TAG = "SmsAlertService"
        private const val CHANNEL_ID = "sms_alert_channel"
        private const val NOTIF_ID = 1010
        const val ACTION_ALERT_SMS = "com.auto_fe.auto_fe.ACTION_ALERT_SMS"
        const val ACTION_STOP = "com.auto_fe.auto_fe.ACTION_STOP_SMS_TTS"
        const val EXTRA_DISPLAY_NAME = "display_name"
        const val EXTRA_BODY = "body"
    }

    private var audioFocusGranted = false
    private var focusRequest: AudioFocusRequest? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createChannelIfNeeded()
        startForeground(NOTIF_ID, buildNotification())

        if (intent?.action == ACTION_ALERT_SMS) {
            val displayName = intent.getStringExtra(EXTRA_DISPLAY_NAME) ?: "số lạ"
            val body = intent.getStringExtra(EXTRA_BODY) ?: ""
            // Nếu đang có cuộc gọi đến/đang gọi, không đọc SMS để tránh cắt cảnh báo cuộc gọi
            if (isInCallOrRinging()) {
                Log.d(TAG, "Phone is ringing or in call; skip SMS TTS to avoid conflict")
                stopForeground(STOP_FOREGROUND_DETACH)
                stopSelf()
            } else {
                speakMessage(displayName, body)
            }
        } else if (intent?.action == ACTION_STOP) {
            try {
                AudioRecorder.getInstance(this).stopSpeaking()
            } catch (_: Exception) {}
            abandonAudioFocus()
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        } else {
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun speakMessage(displayName: String, body: String) {
        try {
            requestAudioFocus()
            val text = if (body.isBlank()) {
                "Bạn có tin nhắn mới từ $displayName."
            } else {
                "Bạn có tin nhắn mới từ $displayName: $body"
            }
            val tts = AudioRecorder.getInstance(this)
            // Không flush để tránh cắt tiếng đang nói (ví dụ cảnh báo cuộc gọi)
            tts.speakNoFlush(text)
            val estimatedMs = (text.length / 10.0 * 1000).toLong().coerceAtLeast(2500)
            android.os.Handler(mainLooper).postDelayed({
                abandonAudioFocus()
                stopForeground(STOP_FOREGROUND_DETACH)
                stopSelf()
            }, estimatedMs)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to speak SMS: ${e.message}")
            abandonAudioFocus()
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }
    }

    private fun isInCallOrRinging(): Boolean {
        return try {
            val tm = getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
            when (tm.callState) {
                TelephonyManager.CALL_STATE_RINGING, TelephonyManager.CALL_STATE_OFFHOOK -> true
                else -> false
            }
        } catch (_: Exception) {
            false
        }
    }

    private fun requestAudioFocus() {
        try {
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val attrs = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
                val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                    .setAudioAttributes(attrs)
                    .setOnAudioFocusChangeListener { }
                    .build()
                focusRequest = req
                audioFocusGranted = audioManager.requestAudioFocus(req) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
            } else {
                @Suppress("DEPRECATION")
                audioFocusGranted = audioManager.requestAudioFocus(
                    null,
                    AudioManager.STREAM_MUSIC,
                    AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK
                ) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
            }
        } catch (e: Exception) {
            Log.w(TAG, "requestAudioFocus failed: ${e.message}")
        }
    }

    private fun abandonAudioFocus() {
        try {
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                focusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
            } else {
                @Suppress("DEPRECATION")
                audioManager.abandonAudioFocus(null)
            }
        } catch (e: Exception) {
            Log.w(TAG, "abandonAudioFocus failed: ${e.message}")
        }
    }

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(getString(R.string.app_name))
            .setContentText("Đọc tin nhắn mới")
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }

    private fun createChannelIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                val channel = NotificationChannel(
                    CHANNEL_ID,
                    "SMS Alert",
                    NotificationManager.IMPORTANCE_LOW
                )
                nm.createNotificationChannel(channel)
            }
        }
    }
}


