package com.auto_fe.auto_fe.automation.phone

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
import androidx.core.app.NotificationCompat
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.audio.AudioRecorder

class CallAlertService : Service() {

    companion object {
        private const val TAG = "CallAlertService"
        private const val CHANNEL_ID = "unknown_call_alert"
        private const val NOTIF_ID = 1009
        const val ACTION_ALERT_UNKNOWN = "com.auto_fe.auto_fe.ACTION_ALERT_UNKNOWN_CALL"
        const val EXTRA_PREV_VOLUME = "prev_volume"
    }

    private var audioFocusGranted = false
    private var focusRequest: AudioFocusRequest? = null
    private var previousRingVolume: Int? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createChannelIfNeeded()
        startForeground(NOTIF_ID, buildNotification())

        if (intent?.action == ACTION_ALERT_UNKNOWN) {
            previousRingVolume = intent.getIntExtra(EXTRA_PREV_VOLUME, -1).takeIf { it >= 0 }
            speakWarningAndStop()
        } else {
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun speakWarningAndStop() {
        try {
            requestAudioFocus()
            // Hạ âm lượng nhạc/phát lại để đảm bảo TTS rõ ràng (duck)
            val tts = AudioRecorder.getInstance(this)
            tts.speak("Bạn đang có cuộc gọi số lạ gọi đến. Hãy đề phòng với các cuộc gọi số lạ.")
            android.os.Handler(mainLooper).postDelayed({
                abandonAudioFocus()
                stopForeground(STOP_FOREGROUND_DETACH)
                stopSelf()
            }, 3500)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to speak warning: ${e.message}")
            abandonAudioFocus()
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
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
            Log.d(TAG, "Audio focus granted: $audioFocusGranted")
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
            .setContentText("Cảnh báo cuộc gọi số lạ")
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
                    "Unknown Call Alert",
                    NotificationManager.IMPORTANCE_LOW
                )
                nm.createNotificationChannel(channel)
            }
        }
    }
}


