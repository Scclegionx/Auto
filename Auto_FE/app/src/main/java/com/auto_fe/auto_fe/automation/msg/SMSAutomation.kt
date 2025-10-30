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

/**
 * SMSAutomation - Enhanced for State Machine
 *
 * THÊM:
 * - Method findExactContactWithPhone() để tìm contact chính xác 100%
 * - Improved logging
 * - Better separation of concerns
 */
class SMSAutomation(private val context: Context) {

    companion object {
        private const val TAG = "SMSAutomation"
    }

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
     * Gửi SMS sử dụng SmsManager API
     */
    fun sendSMS(receiver: String, message: String, callback: SMSCallback) {
        try {
            Log.d(TAG, "sendSMS called with receiver: $receiver, message: $message")

            val phoneNumber = if (isPhoneNumber(receiver)) {
                Log.d(TAG, "Receiver is phone number: $receiver")
                receiver
            } else {
                Log.d(TAG, "Looking up phone number for contact: $receiver")
                val foundNumber = findPhoneNumberByName(receiver)
                Log.d(TAG, "Found phone number: $foundNumber")
                foundNumber
            }

            if (phoneNumber.isEmpty()) {
                Log.e(TAG, "No phone number found for: $receiver")
                callback.onError("Không tìm thấy số điện thoại cho: $receiver")
                return
            }

            Log.d(TAG, "Using phone number: $phoneNumber")

            try {
                val smsManager = SmsManager.getDefault()
                smsManager.sendTextMessage(phoneNumber, null, message, null, null)
                Log.d(TAG, "SMS sent successfully via SmsManager")
                callback.onSuccess()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send SMS via SmsManager: ${e.message}")
                callback.onError("Không thể gửi tin nhắn: ${e.message}")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Exception in sendSMS: ${e.message}", e)
            callback.onError("Lỗi gửi SMS: ${e.message}")
        }
    }

    /**
     * Mở màn hình soạn SMS (không gửi tự động), điền sẵn số và nội dung
     */
    fun openSmsCompose(receiver: String, message: String, callback: SMSCallback) {
        try {
            Log.d(TAG, "openSmsCompose called with receiver: $receiver, message: $message")

            val phoneNumber = if (isPhoneNumber(receiver)) receiver else findPhoneNumberByName(receiver)
            if (phoneNumber.isEmpty()) {
                callback.onError("Không tìm thấy số điện thoại cho: $receiver")
                return
            }

            val intent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("smsto:$phoneNumber")
                putExtra("sms_body", message)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Opened SMS compose UI successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No app available to handle SMS compose")
                callback.onError("Không tìm thấy ứng dụng nhắn tin")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception in openSmsCompose: ${e.message}", e)
            callback.onError("Lỗi mở hộp thoại SMS: ${e.message}")
        }
    }

    /**
     * Kiểm tra xem chuỗi có phải là số điện thoại không
     */
    private fun isPhoneNumber(input: String): Boolean {
        return input.matches(Regex("^[+]?[0-9\\s\\-\\(\\)]+$"))
    }

    /**
     * Tìm số điện thoại từ tên liên hệ (fuzzy search)
     */
    fun findPhoneNumberByName(contactName: String): String {
        Log.d(TAG, "Searching for contact: $contactName")

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "READ_CONTACTS permission not granted")
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

        Log.d(TAG, "No contact found for: $contactName")
        return ""
    }

    /**
     * NEW: Tìm contact chính xác 100% và trả về phone number
     * Dùng cho State Machine khi đã có exact match
     */
    fun findExactContactWithPhone(contactName: String): Pair<String, String>? {
        Log.d(TAG, "Searching for exact contact: $contactName")

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "READ_CONTACTS permission not granted")
            return null
        }

        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )

        // Exact match với =
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
                val displayName = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                Log.d(TAG, "Found exact contact: $displayName with phone: $phoneNumber")
                return Pair(displayName, phoneNumber)
            }
        }

        Log.d(TAG, "No exact contact found for: $contactName")
        return null
    }

    /**
     * Danh sách từ cần bỏ khi tách tên
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

        val similarContacts = mutableSetOf<String>() // Dùng Set để tránh duplicate

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
                    similarContacts.add(displayName)
                }
            }
        }

        return similarContacts.toList()
    }

    /**
     * Kiểm tra khớp 100% với tên mới (DEPRECATED - dùng findExactContactWithPhone)
     */
    @Deprecated("Use findExactContactWithPhone instead", ReplaceWith("findExactContactWithPhone(contactName)"))
    private fun findExactContact(contactName: String): String {
        return findExactContactWithPhone(contactName)?.second ?: ""
    }

    /**
     * Luồng xử lý SMS với xử lý ngoại lệ thông minh
     * Được gọi từ State Machine
     */
    fun sendSMSWithSmartHandling(receiver: String, message: String, callback: SMSConversationCallback) {
        Log.d(TAG, "sendSMSWithSmartHandling called with receiver: $receiver, message: $message")

        // Bước 1: Tách từ thông minh
        val searchWords = smartWordParsing(receiver)
        Log.d(TAG, "Search words after parsing: $searchWords")

        if (searchWords.isEmpty()) {
            Log.d(TAG, "No search words found")
            callback.onError("Không tìm thấy người này trong danh bạ")
            return
        }

        // Bước 2: Tìm kiếm fuzzy
        val similarContacts = findSimilarContacts(searchWords)
        Log.d(TAG, "Similar contacts found: $similarContacts")

        when {
            similarContacts.isEmpty() -> {
                Log.d(TAG, "No similar contacts found")
                callback.onError("Không tìm thấy người này trong danh bạ")
            }

            similarContacts.size == 1 -> {
                val contactName = similarContacts[0]
                Log.d(TAG, "Found 1 similar contact: $contactName, original: $receiver")

                // Kiểm tra xem tên có khớp 100% không
                if (contactName.equals(receiver, ignoreCase = true)) {
                    Log.d(TAG, "Contact name matches exactly, sending directly")
                    // Tên khớp 100% - nhưng vẫn cần confirm qua callback
                    // State Machine sẽ xử lý việc gửi
                    callback.onNeedConfirmation(similarContacts, receiver)
                } else {
                    Log.d(TAG, "Contact name differs, need confirmation")
                    // Tên khác nhau, cần xác nhận
                    callback.onNeedConfirmation(similarContacts, receiver)
                }
            }

            else -> {
                Log.d(TAG, "Found multiple similar contacts: $similarContacts")
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