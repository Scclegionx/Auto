package com.auto_fe.auto_fe.automation.phone

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.content.ContextCompat
import com.auto_fe.auto_fe.base.ConfirmationRequirement
import com.auto_fe.auto_fe.utils.common.SettingsManager
import com.auto_fe.auto_fe.utils.nlp.ContactUtils
import org.json.JSONObject

class PhoneAutomation(private val context: Context) {

    companion object {
        private const val TAG = "PhoneAutomation"
    }

    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     */
    suspend fun executeWithEntities(entities: JSONObject, originalInput: String = ""): String {
        Log.d(TAG, "Executing phone call with entities: $entities")

        // Parse dữ liệu
        val receiver = entities.optString("RECEIVER", "")
        val platform = entities.optString("PLATFORM", "phone").lowercase()

        // Validate
        if (receiver.isEmpty()) {
            throw Exception("Cần chỉ định người nhận cuộc gọi")
        }

        // Kiểm tra setting hỗ trợ nói
        val settingsManager = SettingsManager(context)
        val isSupportSpeakEnabled = settingsManager.isSupportSpeakEnabled()

        // Tìm kiếm liên hệ: Thử exact match trước, nếu không có thì tìm fuzzy
        if (!ContactUtils.isPhoneNumber(receiver)) {
            val exactMatch = ContactUtils.smartFindContact(context, receiver)
            
            if (exactMatch == null) {
                // Không tìm thấy exact match, thử tìm fuzzy
                val similarContacts = ContactUtils.findSimilarContactsWithPhone(context, receiver)
                
                if (similarContacts.isNotEmpty()) {
                    // Tìm thấy các liên hệ tương tự, cần xác nhận
                    val confirmationQuestion = buildSimilarContactsQuestion(similarContacts, receiver)
                    val isMultiple = similarContacts.size > 1
                    throw ConfirmationRequirement(
                        originalInput = originalInput,
                        confirmationQuestion = confirmationQuestion,
                        onConfirmed = {
                            // Nếu chỉ có 1 liên hệ, sử dụng luôn
                            if (!isMultiple) {
                                makeCall(similarContacts[0].name, platform, requireConfirmation = true)
                            } else {
                                // Nếu có nhiều liên hệ, callback này sẽ không được gọi
                                // Thay vào đó, handleConfirmation sẽ xử lý
                                throw Exception("Multiple contacts - should be handled in handleConfirmation")
                            }
                        },
                        isMultipleContacts = isMultiple,
                        actionType = "phone",
                        actionData = platform
                    )
                } else {
                    // Không tìm thấy liên hệ nào
                    throw Exception("Dạ, trong danh bạ chưa có tên này ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại lệnh gọi điện nhé.")
                }
            }
            // Nếu tìm thấy exact match, tiếp tục logic bình thường
        }

        // Routing logic: Gọi điện (chỉ chạy khi đã có exact match hoặc là số điện thoại)
        if (isSupportSpeakEnabled) {
            // Bật hỗ trợ nói: Cần xác nhận trước khi gọi
            val confirmationQuestion = "Dạ, có phải bác muốn $originalInput?"
            throw ConfirmationRequirement(
                originalInput = originalInput,
                confirmationQuestion = confirmationQuestion,
                onConfirmed = {
                    makeCall(receiver, platform, requireConfirmation = true)
                }
            )
        } else {
            // Tắt hỗ trợ nói: Mở màn hình quay số
            return makeCall(receiver, platform, requireConfirmation = false)
        }
    }
    
    /**
     * Xây dựng câu hỏi xác nhận khi tìm thấy các liên hệ tương tự
     */
    private fun buildSimilarContactsQuestion(similarContacts: List<ContactUtils.SimilarContact>, originalInput: String): String {
        return if (similarContacts.size == 1) {
            // 1 liên hệ: Hỏi xác nhận có/không
            "Dạ, con tìm thấy liên hệ ${similarContacts[0].name}. Có phải bác muốn gọi liên hệ này không ạ?"
        } else {
            // Nhiều liên hệ: Hỏi tên cụ thể
            val namesList = similarContacts.take(3).joinToString(", ") { it.name }
            "Dạ, con tìm thấy ${similarContacts.size} liên hệ tương tự: $namesList. Bác muốn gọi liên hệ nào ạ?"
        }
    }

    /**
     * Gọi điện trực tiếp (public để có thể gọi từ AutomationWorkflowManager)
     */
    fun makeCallDirect(receiver: String, platform: String = "phone"): String {
        return makeCall(receiver, platform, requireConfirmation = true)
    }
    
    /**
     * Gọi điện sử dụng Android Intents API (public để có thể gọi từ AutomationWorkflowManager)
     * @param receiver Tên người nhận hoặc số điện thoại
     * @param platform Platform để gọi điện (phone, zalo, etc.)
     * @param requireConfirmation Nếu true, sẽ gọi trực tiếp. Nếu false, chỉ mở dialer
     */
    fun makeCall(receiver: String, platform: String = "phone", requireConfirmation: Boolean = true): String {
        return try {
            Log.d(TAG, "makeCall called with receiver: $receiver, platform: $platform")

            // Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (ContactUtils.isPhoneNumber(receiver)) {
                Log.d(TAG, "Using direct phone number: $receiver")
                receiver
            } else {
                Log.d(TAG, "Looking up contact: $receiver")
                // Thử tìm exact match trước
                val exactMatch = ContactUtils.smartFindContact(context, receiver)
                if (exactMatch != null) {
                    Log.d(TAG, "Found exact match: ${exactMatch.first} with phone: ${exactMatch.second}")
                    exactMatch.second
                } else {
                    // Thử tìm fuzzy
                    val foundNumber = ContactUtils.findPhoneNumberByName(context, receiver)
                    Log.d(TAG, "Found phone number (fuzzy): $foundNumber")
                    foundNumber
                }
            }

            if (phoneNumber.isEmpty()) {
                Log.e(TAG, "No phone number found for: $receiver")
                throw Exception("Dạ, trong danh bạ chưa có tên này ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại lệnh gọi điện nhé.")
            }

            Log.d(TAG, "Attempting to call: $phoneNumber")

            // Nếu không cần xác nhận (tắt hỗ trợ nói), chỉ mở dialer
            if (!requireConfirmation) {
                try {
                    val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                        data = Uri.parse("tel:$phoneNumber")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    if (dialIntent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(dialIntent)
                        Log.d(TAG, "ACTION_DIAL started (support speak off)")
                        return "Dạ, đã mở màn hình quay số. Bác kiểm tra lại số điện thoại và bấm gọi nhé."
                    } else {
                        Log.e(TAG, "No app available to handle ACTION_DIAL")
                        throw Exception("Không tìm thấy ứng dụng gọi điện")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "ACTION_DIAL failed: ${e.message}", e)
                    throw Exception("Lỗi mở quay số: ${e.message}")
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
                            Log.d(TAG, "Calling via default phone app")
                        }
                        "zalo" -> {
                            // Gọi qua Zalo app
                            callIntent.setPackage("com.zing.zalo")
                            Log.d(TAG, "Calling via Zalo app")
                        }
                        else -> {
                            Log.w(TAG, "Unknown platform: $platform, using default")
                        }
                    }
                    
                    Log.d(TAG, "Trying ACTION_CALL for platform: $platform...")
                    if (callIntent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(callIntent)
                        Log.d(TAG, "ACTION_CALL successful - calling directly")
                        return "Dạ, đang thực hiện cuộc gọi ạ."
                    } else {
                        Log.w(TAG, "ACTION_CALL resolveActivity returned null for platform: $platform")
                        throw Exception("Không tìm thấy ứng dụng gọi điện cho platform: $platform")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "ACTION_CALL failed: ${e.message}", e)
                    throw Exception("Lỗi gọi điện: ${e.message}")
                }
            } else {
                Log.d(TAG, "CALL_PHONE permission not granted, trying ACTION_DIAL as fallback")
                // Fallback: Mở dialer nếu không có permission
                try {
                    val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                        data = Uri.parse("tel:$phoneNumber")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    if (dialIntent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(dialIntent)
                        Log.d(TAG, "ACTION_DIAL started as fallback")
                        return "Dạ, đã mở màn hình quay số. Bác kiểm tra lại số điện thoại và bấm gọi nhé."
                    } else {
                        throw Exception("Không tìm thấy ứng dụng gọi điện")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "ACTION_DIAL fallback failed: ${e.message}", e)
                    throw Exception("Lỗi mở quay số: ${e.message}")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "makeCall exception: ${e.message}", e)
            throw Exception("Lỗi gọi điện: ${e.message}")
        }
    }
}
