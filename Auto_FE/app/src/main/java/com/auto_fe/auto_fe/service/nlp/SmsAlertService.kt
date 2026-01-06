package com.auto_fe.auto_fe.automation.msg

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
import android.telephony.TelephonyManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.audio.TTSManager
import kotlinx.coroutines.*

class SmsAlertService : Service() {

    companion object {
        private const val TAG = "SmsAlertService"
        private const val CHANNEL_ID = "sms_alert_channel"
        private const val NOTIF_ID = 3030
        
        const val ACTION_ALERT_SMS = "ACTION_ALERT_SMS"
        const val EXTRA_PHONE_NUMBER = "extra_phone"
        const val EXTRA_BODY = "extra_body"
    }

    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private var audioManager: AudioManager? = null
    private var focusRequest: AudioFocusRequest? = null

    override fun onCreate() {
        super.onCreate()
        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIF_ID, buildNotification("Đang xử lý tin nhắn mới..."))

        if (intent?.action == ACTION_ALERT_SMS) {
            val number = intent.getStringExtra(EXTRA_PHONE_NUMBER) ?: ""
            val body = intent.getStringExtra(EXTRA_BODY) ?: ""
            
            handleIncomingSms(number, body)
        } else {
            stopSelf()
        }
        
        return START_NOT_STICKY
    }

    private fun handleIncomingSms(number: String, body: String) {
        // Nếu đang gọi điện thoại thì không đọc tin nhắn
        if (isInCall()) {
            Log.d(TAG, "In call, skipping SMS alert")
            stopSelf()
            return
        }

        scope.launch {
            // 1. Check Contact Name (Background)
            val contactName = getContactName(number)
            val displayName = if (contactName.isNotEmpty()) contactName else "số lạ"

            withContext(Dispatchers.Main) {
                // 2. Cập nhật thông báo
                val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                nm.notify(NOTIF_ID, buildNotification("Tin nhắn từ $displayName"))

                // 3. Đọc tin nhắn
                speakSms(displayName, body)
            }
        }
    }

    private fun speakSms(sender: String, message: String) {
        requestAudioFocus()

        val textToSpeak = "Bạn có tin nhắn mới từ $sender. Nội dung là: $message"
        TTSManager.getInstance(this).speak(textToSpeak)

        // Tự động tắt service sau khi đọc xong (ước lượng thời gian)
        val duration = (textToSpeak.length / 10.0 * 1000).toLong().coerceAtLeast(3000)
        
        android.os.Handler(mainLooper).postDelayed({
            abandonAudioFocus()
            stopSelf()
        }, duration)
    }

    private suspend fun getContactName(number: String): String = withContext(Dispatchers.IO) {
        var name = ""
        if (number.isBlank()) return@withContext name
        
        val uri = ContactsContract.PhoneLookup.CONTENT_FILTER_URI.buildUpon().appendPath(number).build()
        val projection = arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME)
        
        try {
            val cursor: Cursor? = contentResolver.query(uri, projection, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    name = it.getString(0)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Contact lookup error", e)
        }
        return@withContext name
    }

    private fun isInCall(): Boolean {
        val tm = getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
        return tm.callState != TelephonyManager.CALL_STATE_IDLE
    }

    // --- Audio Focus & Notification boilerplate (Giống CallAlertService) ---
    
    private fun requestAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val attrs = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                .setAudioAttributes(attrs)
                .setOnAudioFocusChangeListener { }
                .build()
            audioManager?.requestAudioFocus(focusRequest!!)
        } else {
            @Suppress("DEPRECATION")
            audioManager?.requestAudioFocus(null, AudioManager.STREAM_MUSIC, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
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
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_mic_white)
            .setContentTitle("Trợ lý tin nhắn")
            .setContentText(text)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                val channel = NotificationChannel(CHANNEL_ID, "SMS Alerts", NotificationManager.IMPORTANCE_LOW)
                nm.createNotificationChannel(channel)
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null
    
    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }
}