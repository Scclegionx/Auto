package com.auto_fe.auto_fe.automation.msg

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.database.ContentObserver
import android.database.Cursor
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.Telephony
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import com.auto_fe.auto_fe.audio.VoiceManager

class SMSObserver(private val context: Context) : ContentObserver(Handler(Looper.getMainLooper())) {
    
    private var audioManager: VoiceManager? = null
    private var lastSmsId: Long = -1
    private var isProcessingSMS = false
    private var isWaitingForResponse = false
    private var pendingSmsId: Long = -1
    private var pendingSmsBody: String = ""
    private var pendingSmsAddress: String = ""
    private val responseHandler = Handler(Looper.getMainLooper())
    private var responseRunnable: Runnable? = null
    private var speechRecognizer: SpeechRecognizer? = null
    
    // Screen state management
    private var isScreenOn = true  // Mặc định là true (giả sử màn hình đang mở)
    private var pendingSMSNotification = false
    private var screenReceiver: BroadcastReceiver? = null
    
    // Sử dụng singleton AudioManager
    init {
        this.audioManager = VoiceManager.getInstance(context)
        // Lấy SMS ID mới nhất để tránh detect SMS cũ khi app khởi động
        initializeLastSmsId()
    }
    
    private fun initializeLastSmsId() {
        try {
            val cursor: Cursor? = context.contentResolver.query(
                Telephony.Sms.CONTENT_URI,
                arrayOf(Telephony.Sms._ID),
                "${Telephony.Sms.TYPE} = ?",
                arrayOf(Telephony.Sms.MESSAGE_TYPE_INBOX.toString()),
                "${Telephony.Sms.DATE} DESC LIMIT 1"
            )
            
            cursor?.use {
                if (it.moveToFirst()) {
                    lastSmsId = it.getLong(it.getColumnIndexOrThrow(Telephony.Sms._ID))
                    Log.d("SMSObserver", "Initialized lastSmsId to: $lastSmsId")
                } else {
                    lastSmsId = -1
                    Log.d("SMSObserver", "No existing SMS found, lastSmsId set to -1")
                }
            }
        } catch (e: Exception) {
            Log.e("SMSObserver", "Error initializing lastSmsId: ${e.message}")
            lastSmsId = -1
        }
    }
    
    override fun onChange(selfChange: Boolean, uri: Uri?) {
        super.onChange(selfChange, uri)
        
        // Chỉ xử lý khi có SMS mới (không phải SMS đã gửi)
        if (uri != null && uri.toString().contains("content://sms")) {
            // Thêm delay để tránh duplicate detection
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                checkForNewSMS()
            }, 500) // Delay 500ms
        }
    }
    
    private fun checkForNewSMS() {
        // Tránh xử lý duplicate
        if (isProcessingSMS) {
            Log.d("SMSObserver", "Already processing SMS, skipping")
            return
        }
        
        try {
            val cursor: Cursor? = context.contentResolver.query(
                Telephony.Sms.CONTENT_URI,
                arrayOf(
                    Telephony.Sms._ID,
                    Telephony.Sms.ADDRESS,
                    Telephony.Sms.BODY,
                    Telephony.Sms.DATE,
                    Telephony.Sms.TYPE
                ),
                "${Telephony.Sms.TYPE} = ?", // Chỉ lấy SMS nhận được
                arrayOf(Telephony.Sms.MESSAGE_TYPE_INBOX.toString()),
                "${Telephony.Sms.DATE} DESC LIMIT 1"
            )
            
            cursor?.use {
                if (it.moveToFirst()) {
                    val smsId = it.getLong(it.getColumnIndexOrThrow(Telephony.Sms._ID))
                    val address = it.getString(it.getColumnIndexOrThrow(Telephony.Sms.ADDRESS))
                    val body = it.getString(it.getColumnIndexOrThrow(Telephony.Sms.BODY))
                    val type = it.getInt(it.getColumnIndexOrThrow(Telephony.Sms.TYPE))
                    
                    // Chỉ xử lý SMS nhận được (MESSAGE_TYPE_INBOX) và chưa từng xử lý
                    if (type == Telephony.Sms.MESSAGE_TYPE_INBOX && smsId != lastSmsId) {
                        Log.d("SMSObserver", "New incoming SMS detected: ID=$smsId, Address=$address")
                        lastSmsId = smsId
                        isProcessingSMS = true
                        handleNewSMS(address, body)
                    } else if (type != Telephony.Sms.MESSAGE_TYPE_INBOX) {
                        Log.d("SMSObserver", "Skipping outgoing SMS: type=$type")
                    } else {
                        Log.d("SMSObserver", "Skipping already processed SMS: ID=$smsId")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("SMSObserver", "Error checking for new SMS: ${e.message}")
            isProcessingSMS = false
        }
    }
    
    private fun handleNewSMS(address: String, body: String) {
        Log.d("SMSObserver", "New SMS from: $address, body: $body")
        
        // Lấy tên contact từ số điện thoại
        val contactName = getContactName(address)
        val displayName = if (contactName.isNotEmpty()) contactName else address
        
        // Lưu thông tin SMS đang chờ
        pendingSmsId = lastSmsId
        pendingSmsBody = body
        pendingSmsAddress = address
        isWaitingForResponse = true
        
        // Kiểm tra màn hình có đang mở không
        Log.d("SMSObserver", "Current screen state: isScreenOn = $isScreenOn")
        if (isScreenOn) {
            Log.d("SMSObserver", "Screen is on, showing notification immediately")
            showSMSNotification(displayName)
        } else {
            Log.d("SMSObserver", "Screen is off, pending notification until user unlocks device")
            pendingSMSNotification = true
        }
    }
    
    private fun showSMSNotification(displayName: String) {
        // Thông báo và đợi user phản hồi
        val notificationMessage = "Bạn có tin nhắn mới từ $displayName. Bạn có muốn tôi đọc nội dung tin nhắn không?"
        
        // Nói thông báo trước
        audioManager?.speak(notificationMessage)
        
        // Đợi nói xong rồi mới bắt đầu voice recognition
        responseHandler.postDelayed({
            startSimpleVoiceRecognition()
        }, 3000) // Đợi 3 giây để nói xong
    }
    
    private fun startSimpleVoiceRecognition() {
        if (!isWaitingForResponse) return
        
        Log.d("SMSObserver", "Starting simple voice recognition...")
        
        try {
            if (!SpeechRecognizer.isRecognitionAvailable(context)) {
                Log.e("SMSObserver", "Speech recognition not available")
                handleVoiceRecognitionError()
                return
            }
            
            // Tạo SpeechRecognizer riêng cho SMS
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
            speechRecognizer?.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: android.os.Bundle?) {
                    Log.d("SMSObserver", "Ready for speech")
                }
                
                override fun onBeginningOfSpeech() {
                    Log.d("SMSObserver", "Beginning of speech")
                }
                
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    Log.d("SMSObserver", "End of speech")
                }
                
                override fun onError(error: Int) {
                    val errorMessage = when (error) {
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Lỗi mạng"
                        SpeechRecognizer.ERROR_NETWORK -> "Lỗi kết nối mạng"
                        SpeechRecognizer.ERROR_AUDIO -> "Lỗi âm thanh"
                        SpeechRecognizer.ERROR_SERVER -> "Lỗi server"
                        SpeechRecognizer.ERROR_CLIENT -> "Lỗi client"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Không nghe thấy giọng nói"
                        SpeechRecognizer.ERROR_NO_MATCH -> "Không nhận diện được"
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Hệ thống nhận diện đang bận"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Không đủ quyền"
                        else -> "Lỗi không xác định: $error"
                    }
                    Log.e("SMSObserver", "Speech recognition error: $errorMessage")
                    handleVoiceRecognitionError()
                }
                
                override fun onResults(results: android.os.Bundle?) {
                    // Kiểm tra xem còn đang chờ phản hồi không
                    if (!isWaitingForResponse) {
                        Log.d("SMSObserver", "Not waiting for response, ignoring speech results")
                        return
                    }
                    
                    // Hủy timeout trước khi xử lý kết quả
                    responseRunnable?.let { responseHandler.removeCallbacks(it) }
                    
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    if (!matches.isNullOrEmpty()) {
                        val spokenText = matches[0]
                        Log.d("SMSObserver", "Speech recognition result: $spokenText")
                        handleUserResponse(spokenText)
                    } else {
                        Log.d("SMSObserver", "No speech recognition results")
                        handleVoiceRecognitionError()
                    }
                }
                
                override fun onPartialResults(partialResults: android.os.Bundle?) {}
                override fun onEvent(eventType: Int, params: android.os.Bundle?) {}
            })
            
            // Bắt đầu nhận diện
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "vi-VN")
                putExtra(RecognizerIntent.EXTRA_PROMPT, "Bạn có muốn đọc tin nhắn không?")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }
            
            speechRecognizer?.startListening(intent)
            
            // Timeout 8 giây (tăng thời gian để tránh race condition)
            responseRunnable = Runnable {
                if (isWaitingForResponse) {
                    Log.d("SMSObserver", "Voice recognition timeout")
                    speechRecognizer?.cancel()
                    handleVoiceRecognitionError()
                }
            }
            responseHandler.postDelayed(responseRunnable!!, 8000)
            
        } catch (e: Exception) {
            Log.e("SMSObserver", "Exception in voice recognition: ${e.message}")
            handleVoiceRecognitionError()
        }
    }
    
    private fun startListeningForResponse(message: String) {
        Log.d("SMSObserver", "Waiting for user response...")
        
        // Đợi lâu hơn để tránh conflict với voice interaction khác
        responseHandler.postDelayed({
            if (isWaitingForResponse) {
                // Kiểm tra xem AudioManager có đang busy không
                if (audioManager == null) {
                    Log.e("SMSObserver", "AudioManager is null")
                    isWaitingForResponse = false
                    return@postDelayed
                }
                
                // Thử voice recognition với fallback
                try {
                    // Sử dụng AudioManager với thông báo tùy chỉnh
                    audioManager?.startVoiceInteractionWithMessage(message, object : com.auto_fe.auto_fe.audio.VoiceManager.AudioManagerCallback {
                        override fun onSpeechResult(spokenText: String) {
                            Log.d("SMSObserver", "Voice recognition result: $spokenText")
                            handleUserResponse(spokenText)
                        }
                        override fun onConfirmationResult(confirmed: Boolean) {}
                        override fun onError(error: String) {
                            Log.e("SMSObserver", "Error in voice recognition: $error")
                            // Fallback: Nếu voice recognition lỗi, dùng manual input
                            handleVoiceRecognitionError()
                        }
                    })
                    
                    // Bắt đầu timeout 3 giây SAU KHI nói xong
                    startTimeoutTimer()
                } catch (e: Exception) {
                    Log.e("SMSObserver", "Exception in voice recognition: ${e.message}")
                    handleVoiceRecognitionError()
                }
            }
        }, 2000) // Đợi 2 giây để tránh conflict
    }
    
    private fun handleVoiceRecognitionError() {
        Log.d("SMSObserver", "Voice recognition failed or timeout - treating as 'no'")
        // Không nói gì = "không" - không đọc tin nhắn
        audioManager?.speak("Đã hủy đọc tin nhắn")
        
        // Hủy timeout
        responseRunnable?.let { responseHandler.removeCallbacks(it) }
        isWaitingForResponse = false
        isProcessingSMS = false // Reset processing flag
    }
    
    private fun startTimeoutTimer() {
        // Timeout sau 5 giây (đợi nói xong mới bắt đầu đếm)
        responseRunnable = Runnable {
            if (isWaitingForResponse) {
                Log.d("SMSObserver", "Timeout waiting for user response - treating as 'no'")
                isWaitingForResponse = false
                isProcessingSMS = false // Reset processing flag
                audioManager?.speak("Đã hủy đọc tin nhắn")
            }
        }
        responseHandler.postDelayed(responseRunnable!!, 5000)
    }
    
    fun handleUserResponse(response: String) {
        Log.d("SMSObserver", "handleUserResponse called with: '$response', isWaitingForResponse: $isWaitingForResponse")
        if (!isWaitingForResponse) {
            Log.d("SMSObserver", "Not waiting for response, ignoring")
            return
        }
        
        // Hủy timeout và speech recognizer
        responseRunnable?.let { responseHandler.removeCallbacks(it) }
        speechRecognizer?.cancel()
        isWaitingForResponse = false
        isProcessingSMS = false // Reset processing flag
        
        val lowerResponse = response.lowercase().trim()
        
        Log.d("SMSObserver", "User response: '$response' -> parsed as: '$lowerResponse'")
        
        when {
            lowerResponse.contains("có") || lowerResponse.contains("đọc") || 
            lowerResponse.contains("yes") || lowerResponse.contains("được") -> {
                // User muốn đọc tin nhắn
                Log.d("SMSObserver", "User wants to read SMS")
                readSMSContent()
                markSMSAsRead()
            }
            lowerResponse.contains("không") || lowerResponse.contains("no") || 
            lowerResponse.contains("thôi") || lowerResponse.contains("không cần") -> {
                // User không muốn đọc
                Log.d("SMSObserver", "User doesn't want to read SMS")
                audioManager?.speak("Đã hủy đọc tin nhắn")
            }
            else -> {
                // Không hiểu phản hồi
                Log.d("SMSObserver", "User response not understood")
                audioManager?.speak("Tôi không hiểu. Vui lòng nói 'có' hoặc 'không'")
            }
        }
    }
    
    private fun readSMSContent() {
        val contactName = getContactName(pendingSmsAddress)
        val displayName = if (contactName.isNotEmpty()) contactName else pendingSmsAddress
        
        audioManager?.speak("Tin nhắn từ $displayName: $pendingSmsBody")
    }
    
    private fun markSMSAsRead() {
        try {
            val uri = Uri.parse("content://sms/$pendingSmsId")
            val values = android.content.ContentValues()
            values.put("read", 1)
            context.contentResolver.update(uri, values, null, null)
            Log.d("SMSObserver", "Marked SMS as read: $pendingSmsId")
        } catch (e: Exception) {
            Log.e("SMSObserver", "Error marking SMS as read: ${e.message}")
        }
    }
    
    private fun getContactName(phoneNumber: String): String {
        try {
            // Thử nhiều cách để tìm contact
            val cleanNumber = phoneNumber.replace("+84", "0").replace(" ", "").replace("-", "")
            
            // Cách 1: Tìm exact match
            var cursor: Cursor? = context.contentResolver.query(
                android.provider.ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                arrayOf(android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME),
                "${android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER} = ?",
                arrayOf(phoneNumber),
                null
            )
            
            cursor?.use {
                if (it.moveToFirst()) {
                    val name = it.getString(it.getColumnIndexOrThrow(android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                    if (name.isNotEmpty()) return name
                }
            }
            
            // Cách 2: Tìm với số đã clean
            cursor = context.contentResolver.query(
                android.provider.ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                arrayOf(android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME),
                "${android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER} = ?",
                arrayOf(cleanNumber),
                null
            )
            
            cursor?.use {
                if (it.moveToFirst()) {
                    val name = it.getString(it.getColumnIndexOrThrow(android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                    if (name.isNotEmpty()) return name
                }
            }
            
            // Cách 3: Tìm với LIKE (partial match)
            cursor = context.contentResolver.query(
                android.provider.ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                arrayOf(android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME),
                "${android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER} LIKE ?",
                arrayOf("%${cleanNumber.takeLast(8)}%"), // Tìm 8 số cuối
                null
            )
            
            cursor?.use {
                if (it.moveToFirst()) {
                    val name = it.getString(it.getColumnIndexOrThrow(android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                    if (name.isNotEmpty()) return name
                }
            }
            
        } catch (e: Exception) {
            Log.e("SMSObserver", "Error getting contact name: ${e.message}")
        }
        return ""
    }
    
    fun startObserving() {
        try {
            // Đăng ký SMS observer
            context.contentResolver.registerContentObserver(
                Telephony.Sms.CONTENT_URI,
                true,
                this
            )
            
            // Đăng ký unlock receiver
            screenReceiver = object : BroadcastReceiver() {
                override fun onReceive(context: Context?, intent: Intent?) {
                    when (intent?.action) {
                        Intent.ACTION_USER_PRESENT -> {
                            Log.d("SMSObserver", "User unlocked device")
                            isScreenOn = true
                            
                            // Nếu có SMS đang chờ thông báo
                            if (pendingSMSNotification) {
                                Log.d("SMSObserver", "Showing pending SMS notification")
                                pendingSMSNotification = false
                                
                                // Lấy thông tin SMS đang chờ
                                val contactName = getContactName(pendingSmsAddress)
                                val displayName = if (contactName.isNotEmpty()) contactName else pendingSmsAddress
                                
                                showSMSNotification(displayName)
                            }
                        }
                        Intent.ACTION_SCREEN_OFF -> {
                            Log.d("SMSObserver", "Screen turned OFF")
                            isScreenOn = false
                        }
                    }
                }
            }
            
            // Đăng ký unlock broadcasts
            val filter = IntentFilter().apply {
                addAction(Intent.ACTION_USER_PRESENT)  // Unlock event
                addAction(Intent.ACTION_SCREEN_OFF)    // Screen off
            }
            context.registerReceiver(screenReceiver, filter)
            
            Log.d("SMSObserver", "Started observing SMS changes and screen state")
        } catch (e: Exception) {
            Log.e("SMSObserver", "Error starting SMS observer: ${e.message}")
        }
    }
    
    fun stopObserving() {
        try {
            // Hủy timeout nếu đang chờ
            responseRunnable?.let { responseHandler.removeCallbacks(it) }
            isWaitingForResponse = false
            
            // Cleanup SpeechRecognizer
            speechRecognizer?.cancel()
            speechRecognizer?.destroy()
            speechRecognizer = null
            
            // Unregister SMS observer
            context.contentResolver.unregisterContentObserver(this)
            
            // Unregister screen state receiver
            screenReceiver?.let { receiver ->
                try {
                    context.unregisterReceiver(receiver)
                } catch (e: Exception) {
                    Log.e("SMSObserver", "Error unregistering screen receiver: ${e.message}")
                }
            }
            screenReceiver = null
            
            Log.d("SMSObserver", "Stopped observing SMS changes and screen state")
        } catch (e: Exception) {
            Log.e("SMSObserver", "Error stopping SMS observer: ${e.message}")
        }
    }
    
    // Method để test manual (debug)
    fun testManualResponse(response: String) {
        Log.d("SMSObserver", "Manual test response: $response")
        handleUserResponse(response)
    }
}
