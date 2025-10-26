package com.auto_fe.auto_fe.core

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.automation.phone.PhoneAutomation
import com.auto_fe.auto_fe.service.NLPService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject

class CommandProcessor(private val context: Context) {
    private val smsAutomation = SMSAutomation(context)
    private val phoneAutomation = PhoneAutomation(context)
    private val nlpService = NLPService(context)

    interface CommandProcessorCallback {
        fun onCommandExecuted(success: Boolean, message: String)
        fun onError(error: String)
        fun onNeedConfirmation(command: String, receiver: String, message: String)
    }

    /**
     * Giải phóng tất cả resources
     */
    fun release() {
        try {
            phoneAutomation.release()
            Log.d("CommandProcessor", "Resources released successfully")
        } catch (e: Exception) {
            Log.e("CommandProcessor", "Error releasing resources: ${e.message}")
        }
    }

    fun processCommand(command: String, callback: CommandProcessorCallback) {
        Log.d("CommandProcessor", "Processing command: $command")
        CoroutineScope(Dispatchers.Main).launch {
            nlpService.sendCommandToServer(command, object : NLPService.NLPServiceCallback {
                override fun onSuccess(response: NLPService.NLPResponse) {
                    Log.d("CommandProcessor", "NLP Response: ${response.rawJson}")
                    // Xử lý dạng 1 (JSON với entities)
                    processCommandFormat1(response.rawJson, callback)
                    
                    // Xử lý dạng 2 (comment lại để dùng dạng 1)
                    // processCommandFormat2(response.rawJson, callback)
                }

                override fun onError(error: String) {
                    Log.e("CommandProcessor", "NLP Error: $error")
                    callback.onError("Lỗi NLP: $error")
                }
            })
        }
    }

    /**
     * Xử lý dạng 1: JSON với entities
     * {
            "text": "Nhắn tin cho cháu Dương là chiều nay 3h qua nhà bà ăn cơm",
            "intent": "send-mess",
            "confidence": 1.0,
            "command": "Gửi tin nhắn",
            "entities": {
                "RECEIVER": "cháu dương",
                "MESSAGE": "chiều nay 3h qua nhà bà ăn cơm",
                "PLATFORM": "sms"
            },
            "value": "chiều nay 3h qua nhà bà ăn cơm",
            "method": "reasoning_engine",
            "processing_time": 0.301513,
            "timestamp": "2025-10-05T11:07:19.436163",
            "reasoning_details": {}
        }
     */
    private fun processCommandFormat1(jsonResponse: String, callback: CommandProcessorCallback) {
        try {
            val json = JSONObject(jsonResponse)
            val command = json.optString("command", "")
            val entities = json.optJSONObject("entities")
            val value = json.optString("value", "")

            if (entities == null) {
                callback.onError("Thiếu thông tin entities")
                return
            }

            val receiver = entities.optString("RECEIVER", "")

            when (command) {
                "Gửi tin nhắn" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("Gửi tin nhắn", receiver, value)
                }
                "call" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("call", receiver, "")
                }
                "search-chrome" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("search-chrome", receiver, "")
                }
                "search-youtube" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("search-youtube", receiver, "")
                }
                "set-alarm" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("set-alarm", receiver, value)
                }
                "add-calendar" -> {
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("add-calendar", receiver, value)
                }
                
                // ========== DEVICE CONTROL COMMANDS ==========
                
                // Camera commands
                "capture-photo" -> {
                    callback.onNeedConfirmation("capture-photo", "photo", "")
                }
                "capture-video" -> {
                    callback.onNeedConfirmation("capture-video", "video", "")
                }
                
                // WiFi commands
                "wifi-enable" -> {
                    callback.onNeedConfirmation("wifi-enable", "enable", "")
                }
                "wifi-disable" -> {
                    callback.onNeedConfirmation("wifi-disable", "disable", "")
                }
                "wifi-toggle" -> {
                    callback.onNeedConfirmation("wifi-toggle", "toggle", "")
                }
                
                // Volume commands
                "volume-increase" -> {
                    callback.onNeedConfirmation("volume-increase", "increase", "")
                }
                "volume-decrease" -> {
                    callback.onNeedConfirmation("volume-decrease", "decrease", "")
                }
                "volume-set" -> {
                    callback.onNeedConfirmation("volume-set", "set", receiver)
                }
                
                // Flash commands
                "flash-enable" -> {
                    callback.onNeedConfirmation("flash-enable", "enable", "")
                }
                "flash-disable" -> {
                    callback.onNeedConfirmation("flash-disable", "disable", "")
                }
                "flash-toggle" -> {
                    callback.onNeedConfirmation("flash-toggle", "toggle", "")
                }
                
                else -> {
                    callback.onError("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ.")
                }
            }
        } catch (e: Exception) {
            callback.onError("Lỗi xử lý dạng 1: ${e.message}")
        }
    }

    /**
     * Xử lý dạng 2: JSON đơn giản
     * {
     *   "output": "command: send-mess  ent: vương  val: chiều đón bà lúc 3h"
     * }
     */
    private fun processCommandFormat2(jsonResponse: String, callback: CommandProcessorCallback) {
        try {
            Log.d("CommandProcessor", "Processing format 2: $jsonResponse")
            val json = JSONObject(jsonResponse)
            val output = json.optString("output", "")

            if (output.isEmpty()) {
                Log.e("CommandProcessor", "Empty output")
                callback.onError("Thiếu thông tin output")
                return
            }

            Log.d("CommandProcessor", "Output: $output")

            // Parse: "command: send-mess  ent: vương  val: chiều đón bà lúc 3h"
            val parts = output.split("  ")
            var command = ""
            var ent = ""
            var value = ""

            for (part in parts) {
                when {
                    part.startsWith("command: ") -> command = part.substring(9)
                    part.startsWith("ent: ") -> ent = part.substring(5)
                    part.startsWith("val: ") -> value = part.substring(5)
                }
            }

            Log.d("CommandProcessor", "Parsed - command: $command, ent: $ent, value: $value")

            when (command) {
                "send-mess" -> {
                    Log.d("CommandProcessor", "Executing SMS command with smart handling")
                    smsAutomation.sendSMSWithSmartHandling(ent, value, object : SMSAutomation.SMSConversationCallback {
                        override fun onSuccess() {
                            Log.d("CommandProcessor", "SMS sent successfully")
                            callback.onCommandExecuted(true, "Đã gửi tin nhắn thành công")
                        }
                        override fun onError(error: String) {
                            Log.e("CommandProcessor", "SMS error: $error")
                            callback.onError(error)
                        }
                        override fun onNeedConfirmation(similarContacts: List<String>, originalName: String) {
                            Log.d("CommandProcessor", "Need confirmation for similar contacts: $similarContacts")
                            val contactList = similarContacts.joinToString(" và ")
                            val message = "Không tìm thấy danh bạ $originalName nhưng tìm được ${similarContacts.size} danh bạ có tên gần giống là $contactList. Liệu bạn có nhầm lẫn tên người gửi không? Nếu nhầm lẫn bạn hãy nói lại tên"
                            callback.onError(message)
                        }
                    })
                }
                "call" -> {
                    Log.d("CommandProcessor", "Parsed call command - contact: $ent")
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("call", ent, "")
                }
                "search-chrome" -> {
                    Log.d("CommandProcessor", "Parsed search-chrome command - query: $ent")
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("search-chrome", ent, "")
                }
                "search-youtube" -> {
                    Log.d("CommandProcessor", "Parsed search-youtube command - query: $ent")
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("search-youtube", ent, "")
                }
                "set-alarm" -> {
                    Log.d("CommandProcessor", "Parsed set-alarm command - time: $ent, message: $value")
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("set-alarm", ent, value)
                }
                "add-calendar" -> {
                    Log.d("CommandProcessor", "Parsed add-calendar command - title: $ent, details: $value")
                    // Parse thành công, trả về data cho FE xử lý
                    callback.onNeedConfirmation("add-calendar", ent, value)
                }
                
                // ========== DEVICE CONTROL COMMANDS ==========
                
                // Camera commands
                "capture-photo" -> {
                    Log.d("CommandProcessor", "Parsed capture-photo command")
                    callback.onNeedConfirmation("capture-photo", "photo", "")
                }
                "capture-video" -> {
                    Log.d("CommandProcessor", "Parsed capture-video command")
                    callback.onNeedConfirmation("capture-video", "video", "")
                }
                
                // WiFi commands
                "wifi-enable" -> {
                    Log.d("CommandProcessor", "Parsed wifi-enable command")
                    callback.onNeedConfirmation("wifi-enable", "enable", "")
                }
                "wifi-disable" -> {
                    Log.d("CommandProcessor", "Parsed wifi-disable command")
                    callback.onNeedConfirmation("wifi-disable", "disable", "")
                }
                "wifi-toggle" -> {
                    Log.d("CommandProcessor", "Parsed wifi-toggle command")
                    callback.onNeedConfirmation("wifi-toggle", "toggle", "")
                }
                
                // Volume commands
                "volume-increase" -> {
                    Log.d("CommandProcessor", "Parsed volume-increase command")
                    callback.onNeedConfirmation("volume-increase", "increase", "")
                }
                "volume-decrease" -> {
                    Log.d("CommandProcessor", "Parsed volume-decrease command")
                    callback.onNeedConfirmation("volume-decrease", "decrease", "")
                }
                "volume-set" -> {
                    Log.d("CommandProcessor", "Parsed volume-set command - value: $ent")
                    callback.onNeedConfirmation("volume-set", "set", ent)
                }
                
                // Flash commands
                "flash-enable" -> {
                    Log.d("CommandProcessor", "Parsed flash-enable command")
                    callback.onNeedConfirmation("flash-enable", "enable", "")
                }
                "flash-disable" -> {
                    Log.d("CommandProcessor", "Parsed flash-disable command")
                    callback.onNeedConfirmation("flash-disable", "disable", "")
                }
                "flash-toggle" -> {
                    Log.d("CommandProcessor", "Parsed flash-toggle command")
                    callback.onNeedConfirmation("flash-toggle", "toggle", "")
                }
                
                else -> {
                    Log.e("CommandProcessor", "Unsupported command: $command")
                    callback.onError("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ.")
                }
            }
        } catch (e: Exception) {
            Log.e("CommandProcessor", "Exception in format 2: ${e.message}")
            callback.onError("Lỗi xử lý dạng 2: ${e.message}")
        }
    }
    
    /**
     * Xử lý khi user nói lại tên người nhận
     */
    fun processSMSWithNewReceiver(newReceiver: String, message: String, callback: CommandProcessorCallback) {
        smsAutomation.sendSMSWithSmartHandling(newReceiver, message, object : SMSAutomation.SMSConversationCallback {
            override fun onSuccess() {
                callback.onCommandExecuted(true, "Đã gửi tin nhắn thành công")
            }
            override fun onError(error: String) {
                callback.onError(error)
            }
            override fun onNeedConfirmation(similarContacts: List<String>, originalName: String) {
                val contactList = similarContacts.joinToString(" và ")
                val message = "Không tìm thấy danh bạ $originalName nhưng tìm được ${similarContacts.size} danh bạ có tên gần giống là $contactList. Liệu bạn có nhầm lẫn tên người gửi không? Nếu nhầm lẫn bạn hãy nói lại tên"
                callback.onError(message)
            }
        })
    }
}