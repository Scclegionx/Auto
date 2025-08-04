package com.auto_fe.core

import android.content.Context
import android.util.Log
import android.widget.Toast
import com.auto_fe.automation.msg.MsgAutomationHelper

class AutomationCommand(private val context: Context) {
    
    companion object {
        private const val TAG = "AutomationCommand"
    }
    
    private lateinit var msgAutomationHelper: MsgAutomationHelper
    // TODO: Thêm các helper khác khi cần
    // private lateinit var callHelper: CallHelper
    // private lateinit var searchHelper: SearchHelper
    
    init {
        msgAutomationHelper = MsgAutomationHelper(context)
        // TODO: Khởi tạo các helper khác
        // callHelper = CallHelper(context)
        // searchHelper = SearchHelper(context)
    }
    
    /**
     * Hàm chung để xử lý tất cả các lệnh automation
     * @param command: Loại lệnh (send-mes, make-call, search-web)
     * @param entities: JSON string chứa thông tin đối tượng
     * @param values: JSON string chứa giá trị cần thực hiện
     */
    fun executeCommand(command: String, entities: String, values: String) {
        Log.d(TAG, "executeCommand: command=$command, entities=$entities, values=$values")
        
        try {
            when (command) {
                Constants.COMMAND_SEND_MESSAGE -> {
                    msgAutomationHelper.sendMessage(command, entities, values)
                }
                Constants.COMMAND_MAKE_CALL -> {
                    // TODO: Implement call functionality
                    // callHelper.makeCall(command, entities, values)
                    Toast.makeText(context, "Tính năng gọi điện chưa được implement", Toast.LENGTH_SHORT).show()
                }
                Constants.COMMAND_SEARCH_WEB -> {
                    // TODO: Implement search functionality
                    // searchHelper.searchWeb(command, entities, values)
                    Toast.makeText(context, "Tính năng tìm kiếm chưa được implement", Toast.LENGTH_SHORT).show()
                }
                else -> {
                    Log.e(TAG, "Unknown command: $command")
                    Toast.makeText(context, "${Constants.ToastMessages.UNSUPPORTED_COMMAND}: $command", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error executing command: ${e.message}", e)
            Toast.makeText(context, "${Constants.ToastMessages.COMMAND_EXECUTION_ERROR} ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    /**
     * Hàm tiện ích để parse JSON entities
     */
    fun getEntityValue(entities: String, key: String, defaultValue: String = ""): String {
        return try {
            val json = org.json.JSONObject(entities)
            json.optString(key, defaultValue)
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing entities: ${e.message}")
            defaultValue
        }
    }
    
    /**
     * Hàm tiện ích để parse JSON values
     */
    fun getValue(values: String, key: String, defaultValue: String = ""): String {
        return try {
            val json = org.json.JSONObject(values)
            json.optString(key, defaultValue)
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing values: ${e.message}")
            defaultValue
        }
    }
} 