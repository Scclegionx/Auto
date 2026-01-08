package com.auto_fe.auto_fe.service.nlp

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
import com.auto_fe.auto_fe.audio.STTManager
import com.auto_fe.auto_fe.audio.TTSManager
import kotlinx.coroutines.*
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import java.util.concurrent.ConcurrentLinkedQueue

data class SmsMessage(
    val displayName: String,
    val body: String,
    val timestamp: Long = System.currentTimeMillis()
)

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

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val smsQueue = ConcurrentLinkedQueue<SmsMessage>()
    private var isProcessingQueue = false
    private var shouldStopProcessing = false
    private var audioFocusGranted = false
    private var focusRequest: AudioFocusRequest? = null
    private lateinit var ttsManager: TTSManager
    private lateinit var sttManager: STTManager

    override fun onCreate() {
        super.onCreate()
        ttsManager = TTSManager.getInstance(this)
        sttManager = STTManager.getInstance(this)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createChannelIfNeeded()
        startForeground(NOTIF_ID, buildNotification("Đang xử lý tin nhắn..."))

        if (intent?.action == ACTION_ALERT_SMS) {
            val displayName = intent.getStringExtra(EXTRA_DISPLAY_NAME) ?: "số lạ"
            val body = intent.getStringExtra(EXTRA_BODY) ?: ""
            
            // Nếu đang có cuộc gọi đến/đang gọi, không đọc SMS để tránh cắt cảnh báo cuộc gọi
            if (isInCallOrRinging()) {
                Log.d(TAG, "Phone is ringing or in call; skip SMS TTS to avoid conflict")
                stopForeground(STOP_FOREGROUND_DETACH)
                stopSelf()
            } else {
                // Thêm vào queue
                val sms = SmsMessage(displayName, body)
                smsQueue.offer(sms)
                Log.d(TAG, "Added SMS to queue from $displayName. Queue size: ${smsQueue.size}")
                
                // Cập nhật notification
                updateNotification("Có ${smsQueue.size} tin nhắn chờ xử lý")
                
                // Bắt đầu xử lý queue nếu chưa đang xử lý
                if (!isProcessingQueue) {
                    processSmsQueue()
                }
            }
        } else if (intent?.action == ACTION_STOP) {
            shouldStopProcessing = true
            try {
                ttsManager.stopSpeaking()
                sttManager.stopListening()
            } catch (_: Exception) {}
            abandonAudioFocus()
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        } else {
            stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun processSmsQueue() {
        if (isProcessingQueue) return
        isProcessingQueue = true
        shouldStopProcessing = false

        scope.launch {
            try {
                while (smsQueue.isNotEmpty() && !shouldStopProcessing) {
                    val sms = smsQueue.poll() ?: break
                    Log.d(TAG, "Processing SMS from ${sms.displayName}")

                    // Cập nhật notification
                    updateNotification("Tin nhắn từ ${sms.displayName}")

                    // Hỏi người dùng có muốn đọc không
                    val shouldRead = askUserForConfirmation(sms.displayName)
                    
                    if (shouldRead && !shouldStopProcessing) {
                        // Đọc nội dung tin nhắn
                        readSmsContent(sms.displayName, sms.body)
                    } else {
                        Log.d(TAG, "User declined to read SMS from ${sms.displayName} or stopped processing")
                        // Nếu người dùng nói không, thông báo và dừng luôn
                        ttsManager.speakAndAwait("Dạ, đã ngừng đọc tin nhắn.")
                        shouldStopProcessing = true
                        break
                    }

                    // Đợi một chút trước khi xử lý tin nhắn tiếp theo
                    delay(500)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing SMS queue", e)
            } finally {
                isProcessingQueue = false
                
                // Nếu queue rỗng hoặc đã dừng, dừng service
                if (smsQueue.isEmpty() || shouldStopProcessing) {
                    abandonAudioFocus()
                    delay(1000) // Đợi một chút trước khi dừng
                    stopForeground(STOP_FOREGROUND_DETACH)
                    stopSelf()
                }
            }
        }
    }

    private suspend fun askUserForConfirmation(sender: String): Boolean = withContext(Dispatchers.Main) {
        requestAudioFocus()
        
        val question = "Dạ, bác có tin nhắn mới từ $sender. Bác có muốn con đọc tin nhắn này không ạ?"
        
        // Nói câu hỏi và chờ nói xong
        ttsManager.speakAndAwait(question)
        
        // Hỏi tối đa 1 lần
        var retryCount = 0
        val maxRetries = 1
        
        while (retryCount < maxRetries && !shouldStopProcessing) {
            val result = listenForUserAnswer()
            
            when {
                result is AnswerResult.Success -> {
                    return@withContext result.value
                }
                result is AnswerResult.Error -> {
                    // Nếu có lỗi (timeout), nói thành tiếng thông báo lỗi
                    ttsManager.speakAndAwait(result.errorMessage)
                    return@withContext false
                }
                else -> {
                    // Không hiểu, hỏi lại
                    retryCount++
                    if (retryCount < maxRetries && !shouldStopProcessing) {
                        ttsManager.speakAndAwait("Dạ, xin lỗi bác, con không hiểu. Bác vui lòng trả lời có hoặc không ạ.")
                    }
                }
            }
        }
        
        // Nếu vẫn không hiểu sau 1 lần, mặc định là không đọc
        false
    }
    
    private sealed class AnswerResult {
        data class Success(val value: Boolean) : AnswerResult()
        data class Error(val errorMessage: String) : AnswerResult()
        object NotUnderstood : AnswerResult()
    }
    
    private suspend fun listenForUserAnswer(): AnswerResult = suspendCancellableCoroutine { continuation ->
        if (shouldStopProcessing) {
            continuation.resumeWith(Result.success(AnswerResult.Success(false)))
            return@suspendCancellableCoroutine
        }
        
        val callback = object : STTManager.STTCallback {
            override fun onSpeechResult(spokenText: String) {
                if (!continuation.isActive || shouldStopProcessing) return
                
                val answer = spokenText.lowercase().trim()
                Log.d(TAG, "User answered: $answer")
                
                // Kiểm tra câu trả lời
                val isYes = answer.contains("có")
                
                val isNo = answer.contains("không")
                
                when {
                    isYes -> continuation.resumeWith(Result.success(AnswerResult.Success(true)))
                    isNo -> continuation.resumeWith(Result.success(AnswerResult.Success(false)))
                    else -> continuation.resumeWith(Result.success(AnswerResult.NotUnderstood)) // Không hiểu, cần hỏi lại
                }
            }

            override fun onError(error: String) {
                if (!continuation.isActive) return
                Log.e(TAG, "STT error: $error")
                // Nếu có lỗi (timeout), trả về error message để nói thành tiếng
                continuation.resumeWith(Result.success(AnswerResult.Error(error)))
            }

            override fun onAudioLevelChanged(level: Int) {
                // Không cần xử lý
            }
        }
        
        sttManager.startListening(callback)
        
        // Xử lý khi coroutine bị hủy
        continuation.invokeOnCancellation {
            sttManager.stopListening()
        }
    }

    private suspend fun readSmsContent(sender: String, message: String) = withContext(Dispatchers.Main) {
        if (shouldStopProcessing) return@withContext
        
        val textToSpeak = "Dạ, nội dung tin nhắn là: $message"
        ttsManager.speakAndAwait(textToSpeak)
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

    private fun buildNotification(text: String = "Đọc tin nhắn mới"): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
    
    private fun updateNotification(text: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIF_ID, buildNotification(text))
    }
    
    override fun onDestroy() {
        scope.cancel()
        smsQueue.clear()
        shouldStopProcessing = true
        try {
            ttsManager.stopSpeaking()
            sttManager.stopListening()
        } catch (_: Exception) {}
        abandonAudioFocus()
        super.onDestroy()
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