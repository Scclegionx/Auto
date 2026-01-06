package com.auto_fe.auto_fe.automation.phone

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.database.Cursor
import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import android.speech.tts.TextToSpeech
import androidx.core.content.ContextCompat
import androidx.core.app.ActivityCompat
import java.util.*
import com.auto_fe.auto_fe.utils.common.SettingsManager

class PhoneAutomation(private val context: Context) {

    private var tts: TextToSpeech? = null

    interface PhoneCallback {
        fun onSuccess()
        fun onError(error: String)
        fun onPermissionRequired() // Callback khi cần permission
    }

    init {
        // Khởi tạo Text-to-Speech cho tiếng Việt
        initTTS()
    }

    /**
     * Kiểm tra và request permission CALL_PHONE
     */
    fun checkAndRequestPermission(callback: PhoneCallback) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CALL_PHONE) 
            != PackageManager.PERMISSION_GRANTED) {
            Log.d("PhoneAutomation", "CALL_PHONE permission not granted, requesting...")
            callback.onPermissionRequired()
        } else {
            Log.d("PhoneAutomation", "CALL_PHONE permission already granted")
            callback.onSuccess() // Gọi callback.onSuccess() khi đã có permission
        }
    }

    /**
     * Gọi điện sử dụng Android Intents API
     * Thử nhiều phương pháp fallback để đảm bảo tương thích tối đa
     * @param receiver Tên người nhận hoặc số điện thoại
     * @param platform Platform để gọi điện (phone, zalo, etc.)
     */
    fun makeCall(receiver: String, platform: String = "phone", callback: PhoneCallback) {
        try {
            Log.d("PhoneAutomation", "makeCall called with receiver: $receiver")

            //Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (isPhoneNumber(receiver)) {
                Log.d("PhoneAutomation", "Using direct phone number: $receiver")
                receiver
            } else {
                Log.d("PhoneAutomation", "Looking up contact: $receiver")
                val foundNumber = findPhoneNumberByName(receiver)
                Log.d("PhoneAutomation", "Found phone number: $foundNumber")
                foundNumber
            }

            if (phoneNumber.isEmpty()) {
                Log.e("PhoneAutomation", "No phone number found for: $receiver")
                val errorMessage = "Không tìm thấy số điện thoại cho $receiver"
                speakError(errorMessage)
                callback.onError(errorMessage)
                return
            }

            Log.d("PhoneAutomation", "Attempting to call: $phoneNumber")
            var success = false

            // Nhánh theo setting: nếu tắt hỗ trợ nói -> dùng ACTION_DIAL (mở UI, không cần quyền)
            val isSupportSpeakEnabled = SettingsManager(context).isSupportSpeakEnabled()
            if (!isSupportSpeakEnabled) {
                try {
                    val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                        data = Uri.parse("tel:$phoneNumber")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    if (dialIntent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(dialIntent)
                        Log.d("PhoneAutomation", "ACTION_DIAL started (support speak off)")
                        callback.onSuccess()
                        return
                    } else {
                        Log.e("PhoneAutomation", "No app available to handle ACTION_DIAL")
                        callback.onError("Không tìm thấy ứng dụng gọi điện (DIAL)")
                        return
                    }
                } catch (e: Exception) {
                    Log.e("PhoneAutomation", "ACTION_DIAL failed: ${e.message}", e)
                    callback.onError("Lỗi mở quay số: ${e.message}")
                    return
                }
            }

            // Gọi trực tiếp dựa trên platform (không qua chooser)
            // Gọi ngay lập tức, không cần user nhấn nút Call
            // Cần permission nguy hiểm, có thể bị từ chối
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.CALL_PHONE)
                == PackageManager.PERMISSION_GRANTED) {
                try {
                    val callIntent = Intent(Intent.ACTION_CALL).apply {
                        data = Uri.parse("tel:$phoneNumber")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    
                    // Set package dựa trên platform
                    when (platform.lowercase()) {
                        "phone" -> {
                            // Gọi qua app gọi điện mặc định
                            callIntent.setPackage("com.android.server.telecom")
                            Log.d("PhoneAutomation", "Calling via default phone app: com.android.phone")
                        }
                        "zalo" -> {
                            // Gọi qua Zalo app
                            callIntent.setPackage("com.zing.zalo")
                            Log.d("PhoneAutomation", "Calling via Zalo app")
                        }
                        else -> {
                            Log.w("PhoneAutomation", "Unknown platform: $platform, using default")
                        }
                    }
                    
                    Log.d("PhoneAutomation", "Trying ACTION_CALL for platform: $platform...")
                    if (callIntent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(callIntent)
                        Log.d("PhoneAutomation", "ACTION_CALL successful - calling directly")
                        success = true
                    } else {
                        Log.w("PhoneAutomation", "ACTION_CALL resolveActivity returned null for platform: $platform")
                    }
                } catch (e: Exception) {
                    Log.e("PhoneAutomation", "ACTION_CALL failed: ${e.message}")
                }
            } else {
                Log.d("PhoneAutomation", "CALL_PHONE permission not granted, skipping direct call")
            }

            // Kết quả cuối cùng
            if (success) {
                callback.onSuccess()
            } else {
                Log.e("PhoneAutomation", "All dialing methods failed")
                val errorMessage = "Không tìm thấy ứng dụng gọi điện"
                speakError(errorMessage)
                callback.onError(errorMessage)
            }

        } catch (e: Exception) {
            Log.e("PhoneAutomation", "makeCall exception: ${e.message}", e)
            val errorMessage = "Lỗi gọi điện: ${e.message}"
            speakError(errorMessage)
            callback.onError(errorMessage)
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
    private fun findPhoneNumberByName(contactName: String): String {
        // Kiểm tra quyền đọc danh bạ
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
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

        return ""
    }

    /**
     * Khởi tạo Text-to-Speech cho tiếng Việt
     */
    private fun initTTS() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale("vi", "VN"))
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("PhoneAutomation", "Vietnamese language not supported, using default")
                    tts?.setLanguage(Locale.getDefault())
                } else {
                    Log.d("PhoneAutomation", "Vietnamese TTS initialized successfully")
                }
            } else {
                Log.e("PhoneAutomation", "TTS initialization failed")
            }
        }
    }

    /**
     * Nói lỗi bằng tiếng Việt
     */
    private fun speakError(errorMessage: String) {
        try {
            Log.d("PhoneAutomation", "Speaking error: $errorMessage")
            tts?.speak(errorMessage, TextToSpeech.QUEUE_FLUSH, null, "ERROR_SPEECH")
        } catch (e: Exception) {
            Log.e("PhoneAutomation", "Failed to speak error: ${e.message}")
        }
    }

    /**
     * Giải phóng tài nguyên TTS
     */
    fun release() {
        try {
            tts?.stop()
            tts?.shutdown()
            tts = null
            Log.d("PhoneAutomation", "TTS resources released")
        } catch (e: Exception) {
            Log.e("PhoneAutomation", "Error releasing TTS: ${e.message}")
        }
    }
}