package com.auto_fe.accessibility

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONObject

class AutoAccessibilityService : AccessibilityService() {
    
    companion object {
        private const val TAG = "AutoAccessibilityService"
        private var instance: AutoAccessibilityService? = null
        
        fun getInstance(): AutoAccessibilityService? = instance
    }
    
    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        
        val info = AccessibilityServiceInfo().apply {
            eventTypes = AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED or 
                        AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED or
                        AccessibilityEvent.TYPE_VIEW_CLICKED
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            flags = AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS or
                   AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS
            notificationTimeout = 100
        }
        serviceInfo = info
        
        Log.d(TAG, "Accessibility Service connected")
    }
    
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event?.let {
            Log.d(TAG, "Event: ${it.eventType} - Package: ${it.packageName}")
        }
    }
    
    override fun onInterrupt() {
        Log.d(TAG, "Accessibility Service interrupted")
    }
    
    override fun onDestroy() {
        super.onDestroy()
        instance = null
        Log.d(TAG, "Accessibility Service destroyed")
    }
    
    // Hàm chính để xử lý gửi tin nhắn
    fun sendMessage(command: String, entities: String, values: String) {
        Log.d(TAG, "sendMessage called: command=$command, entities=$entities, values=$values")
        
        try {
            when (command) {
                "send-mes" -> {
                    val entitiesJson = JSONObject(entities)
                    val valuesJson = JSONObject(values)
                    
                    val contactName = entitiesJson.optString("ent", "")
                    val messageText = valuesJson.optString("val", "")
                    
                    if (contactName.isNotEmpty() && messageText.isNotEmpty()) {
                        sendMessageToContact(contactName, messageText)
                    } else {
                        Log.e(TAG, "Invalid entities or values")
                    }
                }
                else -> {
                    Log.e(TAG, "Unknown command: $command")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in sendMessage: ${e.message}")
        }
    }
    
    private fun sendMessageToContact(contactName: String, messageText: String) {
        Log.d(TAG, "Sending message to $contactName: $messageText")
        
        try {
            // Bước 1: Mở app tin nhắn
            openMessagesApp()
            
            // Bước 2: Tìm và click vào nút tạo tin nhắn mới
            findAndClickNewMessageButton()
            
            // Bước 3: Tìm contact trong danh bạ
            findAndSelectContact(contactName)
            
            // Bước 4: Nhập tin nhắn
            enterMessageText(messageText)
            
            // Bước 5: Gửi tin nhắn
            sendMessage()
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in sendMessageToContact: ${e.message}")
        }
    }
    
    private fun openMessagesApp() {
        Log.d(TAG, "Opening Messages app")
        
        val intent = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_APP_MESSAGING)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK
        }
        
        try {
            startActivity(intent)
            Thread.sleep(2000) // Đợi app mở
        } catch (e: Exception) {
            Log.e(TAG, "Error opening Messages app: ${e.message}")
        }
    }
    
    private fun findAndClickNewMessageButton() {
        Log.d(TAG, "Finding new message button")
        
        val rootNode = rootInActiveWindow ?: return
        
        // Tìm nút tạo tin nhắn mới (có thể là FAB hoặc nút "+")
        val newMessageButton = findNodeByText(rootNode, "New message") ?:
                              findNodeByText(rootNode, "Compose") ?:
                              findNodeByText(rootNode, "+") ?:
                              findNodeByDescription(rootNode, "New message") ?:
                              findNodeByDescription(rootNode, "Compose")
        
        newMessageButton?.let {
            Log.d(TAG, "Found new message button, clicking...")
            it.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            Thread.sleep(1000)
        } ?: Log.e(TAG, "New message button not found")
    }
    
    private fun findAndSelectContact(contactName: String) {
        Log.d(TAG, "Finding contact: $contactName")
        
        val rootNode = rootInActiveWindow ?: return
        
        // Tìm ô nhập tên contact
        val contactInput = findNodeByText(rootNode, "To") ?:
                          findNodeByDescription(rootNode, "To") ?:
                          findNodeByClassName(rootNode, "android.widget.EditText")
        
                 contactInput?.let {
             Log.d(TAG, "Found contact input, entering contact name...")
             it.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
             val arguments = Bundle()
             arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, contactName)
             it.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
             Thread.sleep(1000)
            
            // Tìm và click vào contact trong danh sách
            val contactInList = findNodeByText(rootNode, contactName)
            contactInList?.let { contact ->
                Log.d(TAG, "Found contact in list, selecting...")
                contact.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                Thread.sleep(1000)
            } ?: Log.e(TAG, "Contact not found in list")
        } ?: Log.e(TAG, "Contact input not found")
    }
    
    private fun enterMessageText(messageText: String) {
        Log.d(TAG, "Entering message text: $messageText")
        
        val rootNode = rootInActiveWindow ?: return
        
        // Tìm ô nhập tin nhắn
        val messageInput = findNodeByText(rootNode, "Text message") ?:
                          findNodeByDescription(rootNode, "Text message") ?:
                          findNodeByClassName(rootNode, "android.widget.EditText")
        
                 messageInput?.let {
             Log.d(TAG, "Found message input, entering text...")
             it.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
             val arguments = Bundle()
             arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, messageText)
             it.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
             Thread.sleep(500)
         } ?: Log.e(TAG, "Message input not found")
    }
    
    private fun sendMessage() {
        Log.d(TAG, "Sending message")
        
        val rootNode = rootInActiveWindow ?: return
        
        // Tìm nút gửi tin nhắn
        val sendButton = findNodeByText(rootNode, "Send") ?:
                        findNodeByDescription(rootNode, "Send") ?:
                        findNodeByText(rootNode, "Send message")
        
        sendButton?.let {
            Log.d(TAG, "Found send button, clicking...")
            it.performAction(AccessibilityNodeInfo.ACTION_CLICK)
            Log.d(TAG, "Message sent successfully!")
        } ?: Log.e(TAG, "Send button not found")
    }
    
    // Helper functions để tìm node
    private fun findNodeByText(rootNode: AccessibilityNodeInfo, text: String): AccessibilityNodeInfo? {
        for (i in 0 until rootNode.childCount) {
            val child = rootNode.getChild(i) ?: continue
            if (child.text?.toString()?.contains(text, ignoreCase = true) == true) {
                return child
            }
            val result = findNodeByText(child, text)
            if (result != null) return result
        }
        return null
    }
    
    private fun findNodeByDescription(rootNode: AccessibilityNodeInfo, description: String): AccessibilityNodeInfo? {
        for (i in 0 until rootNode.childCount) {
            val child = rootNode.getChild(i) ?: continue
            if (child.contentDescription?.toString()?.contains(description, ignoreCase = true) == true) {
                return child
            }
            val result = findNodeByDescription(child, description)
            if (result != null) return result
        }
        return null
    }
    
    private fun findNodeByClassName(rootNode: AccessibilityNodeInfo, className: String): AccessibilityNodeInfo? {
        for (i in 0 until rootNode.childCount) {
            val child = rootNode.getChild(i) ?: continue
            if (child.className?.toString() == className) {
                return child
            }
            val result = findNodeByClassName(child, className)
            if (result != null) return result
        }
        return null
    }
} 