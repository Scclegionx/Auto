package com.auto_fe.auto_fe.automation.msg

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.database.Cursor
import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import android.telephony.SmsManager
import androidx.core.content.ContextCompat

class SMSAutomation(private val context: Context) {
    
    interface SMSCallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    interface SMSConversationCallback {
        fun onSuccess()
        fun onError(error: String)
        fun onNeedConfirmation(similarContacts: List<String>, originalName: String)
    }
    
    /**
     * Gửi SMS sử dụng Android Intents API
     * @param receiver Tên người nhận hoặc số điện thoại
     * @param message Nội dung tin nhắn
     */
    fun sendSMS(receiver: String, message: String, callback: SMSCallback) {
        try {
            Log.d("SMSAutomation", "sendSMS called with receiver: $receiver, message: $message")
            
            // Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (isPhoneNumber(receiver)) {
                Log.d("SMSAutomation", "Receiver is phone number: $receiver")
                receiver
            } else {
                Log.d("SMSAutomation", "Looking up phone number for contact: $receiver")
                val foundNumber = findPhoneNumberByName(receiver)
                Log.d("SMSAutomation", "Found phone number: $foundNumber")
                foundNumber
            }
            
            if (phoneNumber.isEmpty()) {
                Log.e("SMSAutomation", "No phone number found for: $receiver")
                callback.onError("Không tìm thấy số điện thoại cho: $receiver")
                return
            }
            
            Log.d("SMSAutomation", "Using phone number: $phoneNumber")
            
            // Gửi SMS trực tiếp bằng SmsManager API
            try {
                val smsManager = SmsManager.getDefault()
                smsManager.sendTextMessage(phoneNumber, null, message, null, null)
                Log.d("SMSAutomation", "SMS sent successfully via SmsManager")
                callback.onSuccess()
            } catch (e: Exception) {
                Log.e("SMSAutomation", "Failed to send SMS via SmsManager: ${e.message}")
                callback.onError("Không thể gửi tin nhắn: ${e.message}")
            }
            
        } catch (e: Exception) {
            Log.e("SMSAutomation", "Exception in sendSMS: ${e.message}", e)
            callback.onError("Lỗi gửi SMS: ${e.message}")
        }
    }
    
    /**
     * Kiểm tra xem chuỗi có phải là số điện thoại không
     */
    private fun isPhoneNumber(input: String): Boolean {
        return input.matches(Regex("^[+]?[0-9\\s\\-\\(\\)]+$"))
    }
    
    /**
     * Tìm số điện thoại từ tên liên hệ
     */
    fun findPhoneNumberByName(contactName: String): String {
        Log.d("SMSAutomation", "Searching for contact: $contactName")
        
        // Kiểm tra quyền đọc danh bạ
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) 
            != PackageManager.PERMISSION_GRANTED) {
            Log.e("SMSAutomation", "READ_CONTACTS permission not granted")
            return ""
        }
        
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )
        
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$contactName%")
        
        val cursor: Cursor? = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )
        
        cursor?.use {
            if (it.moveToFirst()) {
                val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                return phoneNumber
            }
        }
        
        Log.d("SMSAutomation", "No contact found for: $contactName")
        return ""
    }
    
    /**
     * Danh sách từ cần bỏ khi tách tên (hardcode)
     */
    private val excludedWords = setOf(
        "cháu", "anh", "chị", "em", "bác", "cô", "dì", "thầy", "cô giáo",
        "bà", "ông", "nội", "ngoại", "cậu", "mợ", "dượng", "ba",
        "con", "chú", "bác", "cô", "dì", "thầy", "cô giáo", "bạn", "bạn bè"
    )
    
    /**
     * Tách từ thông minh - bỏ các từ không phải tên riêng
     */
    private fun smartWordParsing(fullName: String): List<String> {
        val words = fullName.trim().split("\\s+".toRegex())
        return words.filter { word ->
            val cleanWord = word.lowercase().trim()
            cleanWord.isNotEmpty() && !excludedWords.contains(cleanWord)
        }
    }
    
    /**
     * Tìm kiếm danh bạ fuzzy - tìm tất cả danh bạ chứa từ khóa
     */
    private fun findSimilarContacts(searchWords: List<String>): List<String> {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) 
            != PackageManager.PERMISSION_GRANTED) {
            return emptyList()
        }
        
        val similarContacts = mutableListOf<String>()
        
        for (word in searchWords) {
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
            )
            
            val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf("%$word%")
            
            val cursor: Cursor? = context.contentResolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                null
            )
            
            cursor?.use {
                while (it.moveToNext()) {
                    val displayName = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                    if (!similarContacts.contains(displayName)) {
                        similarContacts.add(displayName)
                    }
                }
            }
        }
        
        return similarContacts
    }
    
    /**
     * Kiểm tra khớp 100% với tên mới
     */
    private fun findExactContact(contactName: String): String {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) 
            != PackageManager.PERMISSION_GRANTED) {
            return ""
        }
        
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )
        
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} = ?"
        val selectionArgs = arrayOf(contactName)
        
        val cursor: Cursor? = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )
        
        cursor?.use {
            if (it.moveToFirst()) {
                val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                return phoneNumber
            }
        }
        
        return ""
    }
    
    /**
     * Luồng xử lý SMS với xử lý ngoại lệ thông minh
     */
    fun sendSMSWithSmartHandling(receiver: String, message: String, callback: SMSConversationCallback) {
        Log.d("SMSAutomation", "sendSMSWithSmartHandling called with receiver: $receiver, message: $message")
        
        // Bước 1: Tách từ thông minh
        val searchWords = smartWordParsing(receiver)
        Log.d("SMSAutomation", "Search words after parsing: $searchWords")
        
        if (searchWords.isEmpty()) {
            Log.d("SMSAutomation", "No search words found")
            callback.onError("Không tìm thấy người này trong danh bạ")
            return
        }
        
        // Bước 2: Tìm kiếm fuzzy
        val similarContacts = findSimilarContacts(searchWords)
        Log.d("SMSAutomation", "Similar contacts found: $similarContacts")
        
        when {
            similarContacts.isEmpty() -> {
                Log.d("SMSAutomation", "No similar contacts found")
                callback.onError("Không tìm thấy người này trong danh bạ")
            }
            
            similarContacts.size == 1 -> {
                val contactName = similarContacts[0]
                Log.d("SMSAutomation", "Found 1 similar contact: $contactName, original: $receiver")
                
                // Kiểm tra xem tên có khớp 100% không
                if (contactName.lowercase() == receiver.lowercase()) {
                    Log.d("SMSAutomation", "Contact name matches exactly, sending directly")
                    // Tên khớp 100%, gửi trực tiếp
                    val phoneNumber = findExactContact(contactName)
                    if (phoneNumber.isNotEmpty()) {
                        sendSMSDirect(phoneNumber, message, callback)
                    } else {
                        callback.onError("Không tìm thấy người này trong danh bạ")
                    }
                } else {
                    Log.d("SMSAutomation", "Contact name differs, need confirmation")
                    // Tên khác nhau, cần xác nhận
                    callback.onNeedConfirmation(similarContacts, receiver)
                }
            }
            
            else -> {
                Log.d("SMSAutomation", "Found multiple similar contacts: $similarContacts")
                callback.onNeedConfirmation(similarContacts, receiver)
            }
        }
    }
    
    /**
     * Gửi SMS trực tiếp với số điện thoại
     */
    private fun sendSMSDirect(phoneNumber: String, message: String, callback: SMSConversationCallback) {
        sendSMS(phoneNumber, message, object : SMSCallback {
            override fun onSuccess() {
                callback.onSuccess()
            }
            override fun onError(error: String) {
                callback.onError(error)
            }
        })
    }
    
}
