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
                    smsAutomation.sendSMSWithSmartHandling(receiver, value, object : SMSAutomation.SMSConversationCallback {
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
                "call" -> {
                    phoneAutomation.makeCall(receiver, object : PhoneAutomation.PhoneCallback {
                        override fun onSuccess() {
                            callback.onCommandExecuted(true, "Đã gọi điện thành công")
                        }
                        override fun onError(error: String) {
                            callback.onError("Lỗi gọi điện: $error")
                        }
                    })
                }
                else -> {
                    callback.onError("Không hỗ trợ lệnh: $command")
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
                    Log.d("CommandProcessor", "Executing call command")
                    phoneAutomation.makeCall(ent, object : PhoneAutomation.PhoneCallback {
                        override fun onSuccess() {
                            Log.d("CommandProcessor", "Call initiated successfully")
                            val successMessage = "Đã gọi điện thành công"
                            callback.onCommandExecuted(true, successMessage)
                        }
                        override fun onError(error: String) {
                            Log.e("CommandProcessor", "Call error: $error")
                            callback.onError("Lỗi gọi điện: $error")
                        }
                    })
                }
                else -> {
                    Log.e("CommandProcessor", "Unsupported command: $command")
                    callback.onError("Không hỗ trợ lệnh: $command")
                }
            }
        } catch (e: Exception) {
            Log.e("CommandProcessor", "Exception in format 2: ${e.message}")
            callback.onError("Lỗi xử lý dạng 2: ${e.message}")
        }
    }
}