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
import android.util.Log
import com.auto_fe.auto_fe.audio.TTSManager
import com.auto_fe.auto_fe.service.nlp.SmsAlertService
import android.provider.ContactsContract.CommonDataKinds.Phone.CONTENT_URI
import android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
import android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER

class SMSReceiver(private val context: Context) : ContentObserver(Handler(Looper.getMainLooper())) {
    
    private var ttsManager: TTSManager? = null
    private var lastSmsId: Long = -1
    private var isProcessingSMS = false
    private var pendingSmsId: Long = -1
    private var pendingSmsBody: String = ""
    private var pendingSmsAddress: String = ""
    
    // Screen state management
    private var isScreenOn = true  // Mặc định là true (giả sử màn hình đang mở)
    private var pendingSMSNotification = false
    private var screenReceiver: BroadcastReceiver? = null
    
    // Sử dụng singleton TTSManager
    init {
        this.ttsManager = TTSManager.getInstance(context)
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
        
        // Lấy tên contact từ số điện thoại; nếu không có, xưng 'số lạ' thay vì đọc dãy số
        val contactName = getContactName(address)
        val displayName = if (contactName.isNotEmpty()) contactName else "số lạ"

        // Lưu thông tin SMS để dùng sau (cho trường hợp unlock)
        pendingSmsId = lastSmsId
        pendingSmsBody = body
        pendingSmsAddress = address

        // Nếu màn hình tắt, đánh dấu để hiển thị khi unlock
        if (!isScreenOn) {
            Log.d("SMSObserver", "Screen is off, marking SMS for notification when unlocked")
            pendingSMSNotification = true
            isProcessingSMS = false
            return
        }

        // Màn hình đang bật, gửi sang foreground service để đọc ngay
        try {
            val intent = Intent(context, SmsAlertService::class.java).apply {
                action = SmsAlertService.ACTION_ALERT_SMS
                putExtra(SmsAlertService.EXTRA_DISPLAY_NAME, displayName)
                putExtra(SmsAlertService.EXTRA_BODY, body)
            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        } catch (e: Exception) {
            Log.e("SMSObserver", "Failed to start SmsAlertService: ${e.message}")
        }

        // Reset state để tránh chồng xử lý
        isProcessingSMS = false
    }
    
    private fun showSMSNotification(displayName: String) {
        val intent = Intent(context, SmsAlertService::class.java).apply {
            action = SmsAlertService.ACTION_ALERT_SMS
            putExtra(SmsAlertService.EXTRA_DISPLAY_NAME, displayName)
            putExtra(SmsAlertService.EXTRA_BODY, pendingSmsBody)
        }
        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        } catch (e: Exception) {
            Log.e("SMSObserver", "Failed to start SmsAlertService (showSMSNotification): ${e.message}")
        }
    }
    
    private fun getContactName(phoneNumber: String): String {
        try {
            val cleanNumber = phoneNumber.replace("+84", "0").replace(" ", "").replace("-", "")
            
            // Tìm exact match
            var cursor: Cursor? = context.contentResolver.query(
                CONTENT_URI,
                arrayOf(DISPLAY_NAME),
                "${NUMBER} = ?",
                arrayOf(phoneNumber),
                null
            )
            
            cursor?.use {
                if (it.moveToFirst()) {
                    val name = it.getString(it.getColumnIndexOrThrow(DISPLAY_NAME))
                    if (name.isNotEmpty()) return name
                }
            }

            cursor = context.contentResolver.query(
                CONTENT_URI,
                arrayOf(DISPLAY_NAME),
                "${NUMBER} = ?",
                arrayOf(cleanNumber),
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
}