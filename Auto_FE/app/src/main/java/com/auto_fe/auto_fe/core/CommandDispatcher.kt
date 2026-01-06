package com.auto_fe.auto_fe.core

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.base.AutomationTask
import com.auto_fe.auto_fe.service.nlp.NLPService
import org.json.JSONObject

class CommandDispatcher(private val context: Context) : AutomationTask {
    
    companion object {
        private const val TAG = "CommandDispatcher"
    }
    
    private val nlpService = NLPService(context)
    
    override suspend fun execute(input: String): String {
        Log.d(TAG, "Dispatching command: $input")
        
        // 1. Gửi text string hoàn chỉnh lên NLP
        val responseJson = nlpService.analyzeText(input)
        
        // 2. Parse response
        val command = responseJson.optString("command", "")
        val intent = responseJson.optString("intent", "")
        val entities = responseJson.optJSONObject("entities") ?: JSONObject()
        
        // 3. Routing logic - Dựa vào command để gọi đúng class Automation
        val finalCommand = if (command.isNotEmpty()) command else intent
        
        return when (finalCommand) {
            // "send-mess" -> {
            //     val sms = SMSAutomation(context)
            //     sms.executeWithEntities(entities)
            // }
            "set-alarm" -> {
                val alarm = AlarmAutomation(context)
                alarm.executeWithEntities(entities)
            }
            // TODO: Thêm các command khác khi refactor
            else -> {
                throw Exception("Tôi hiểu bạn nói '$input' nhưng chưa hỗ trợ lệnh này")
            }
        }
    }
}

