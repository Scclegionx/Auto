package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.models.ChatMessage
import com.auto_fe.auto_fe.network.ApiConfig
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.disposables.Disposable
import org.json.JSONObject
import ua.naiksoftware.stomp.Stomp
import ua.naiksoftware.stomp.StompClient
import ua.naiksoftware.stomp.dto.LifecycleEvent
import ua.naiksoftware.stomp.dto.StompHeader

/**
 * WebSocket Manager cho real-time chat
 * Sử dụng STOMP protocol over SockJS
 */
class WebSocketManager {
    
    private var stompClient: StompClient? = null
    private val compositeDisposable = CompositeDisposable()
    private var isConnected = false
    
    companion object {
        private const val TAG = "WebSocketManager"
        private const val WS_ENDPOINT = "/ws/chat"
        
        // WebSocket base URL (không có /api)
        private fun getWebSocketBaseUrl(): String {
            // Lấy base URL và remove /api nếu có
            val baseUrl = ApiConfig.BASE_URL.removeSuffix("/api")
            return baseUrl
        }
    }
    
    /**
     * Kết nối WebSocket với JWT token
     */
    fun connect(accessToken: String, onConnected: () -> Unit, onError: (String) -> Unit) {
        if (isConnected) {
            Log.d(TAG, "WebSocket already connected")
            onConnected()
            return
        }
        
        try {
            // Thêm token vào URL như query parameter
            val baseUrl = getWebSocketBaseUrl()
            // Đổi http thành ws cho WebSocket
            val wsBaseUrl = baseUrl.replace("http://", "ws://").replace("https://", "wss://")
            val wsUrl = "${wsBaseUrl}${WS_ENDPOINT}?access_token=$accessToken"
            Log.d(TAG, "WebSocket Base URL: $wsBaseUrl")
            Log.d(TAG, "Full WebSocket URL: $wsUrl")
            Log.d(TAG, "Token preview: ${accessToken.take(20)}...")
            
            // Tạo STOMP client với WebSocket thuần (không dùng SockJS)
            stompClient = Stomp.over(Stomp.ConnectionProvider.OKHTTP, wsUrl)
            
            // Lắng nghe lifecycle events
            val lifecycleDisposable = stompClient?.lifecycle()?.subscribe { lifecycleEvent ->
                when (lifecycleEvent.type) {
                    LifecycleEvent.Type.OPENED -> {
                        Log.d(TAG, "WebSocket connection opened")
                        isConnected = true
                        onConnected()
                    }
                    LifecycleEvent.Type.CLOSED -> {
                        Log.d(TAG, "WebSocket connection closed")
                        isConnected = false
                    }
                    LifecycleEvent.Type.ERROR -> {
                        Log.e(TAG, "WebSocket error: ${lifecycleEvent.exception?.message}")
                        Log.e(TAG, "Error details: ", lifecycleEvent.exception)
                        isConnected = false
                        onError(lifecycleEvent.exception?.message ?: "Unknown error")
                    }
                    else -> {
                        Log.d(TAG, "WebSocket event: ${lifecycleEvent.type}")
                    }
                }
            }
            
            lifecycleDisposable?.let { compositeDisposable.add(it) }
            
            // Kết nối (không cần headers vì token đã ở URL)
            stompClient?.connect()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect WebSocket", e)
            onError(e.message ?: "Connection failed")
        }
    }
    
    /**
     * Subscribe để nhận tin nhắn mới trong queue của user
     * Backend gửi đến: /user/{userEmail}/queue/messages
     */
    fun subscribeToUserMessages(
        userEmail: String,
        onMessageReceived: (ChatMessage) -> Unit
    ): Disposable? {
        if (!isConnected) {
            Log.e(TAG, "Cannot subscribe - WebSocket not connected")
            return null
        }
        
        val destination = "/user/$userEmail/queue/messages"
        Log.d(TAG, "Subscribing to: $destination")
        
        val disposable = stompClient?.topic(destination)?.subscribe({ stompMessage ->
            try {
                Log.d(TAG, "Raw message received: ${stompMessage.payload}")
                val json = JSONObject(stompMessage.payload)
                
                val message = ChatMessage(
                    id = json.getLong("id"),
                    chatId = json.getLong("chatId"),
                    senderId = json.getLong("senderId"),
                    senderName = json.optString("senderName", null),
                    senderAvatar = json.optString("senderAvatar", null),
                    content = json.getString("content"),
                    isRead = json.optBoolean("isRead", false),
                    readAt = json.optString("readAt", null),
                    createdAt = json.optString("createdAt", null)
                )
                
                Log.d(TAG, "Parsed message: id=${message.id}, chatId=${message.chatId}, content=${message.content}")
                onMessageReceived(message)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to parse message: ${stompMessage.payload}", e)
            }
        }, { error ->
            Log.e(TAG, "Subscribe error: ${error.message}", error)
        })
        
        disposable?.let { 
            compositeDisposable.add(it)
            Log.d(TAG, "Successfully subscribed to $destination")
        }
        return disposable
    }
    
    /**
     * Subscribe to a chat topic to receive real-time messages
     */
    fun subscribeToTopic(topic: String, onMessageReceived: (ChatMessage) -> Unit): Disposable? {
        if (!isConnected) {
            Log.e(TAG, "Cannot subscribe - WebSocket not connected")
            return null
        }
        
        Log.d(TAG, "📡 Subscribing to topic: $topic")
        
        val disposable = stompClient?.topic(topic)?.subscribe({ stompMessage ->
            try {
                Log.d(TAG, "📨 Message received from topic: ${stompMessage.payload}")
                val json = JSONObject(stompMessage.payload)
                
                val message = ChatMessage(
                    id = json.getLong("id"),
                    chatId = json.getLong("chatId"),
                    senderId = json.getLong("senderId"),
                    senderName = json.optString("senderName", null),
                    senderAvatar = json.optString("senderAvatar", null),
                    content = json.getString("content"),
                    messageType = json.optString("messageType", "TEXT"),
                    attachmentUrl = json.optString("attachmentUrl", null),
                    attachmentName = json.optString("attachmentName", null),
                    attachmentType = json.optString("attachmentType", null),
                    attachmentSize = if (json.has("attachmentSize") && !json.isNull("attachmentSize")) {
                        json.getLong("attachmentSize")
                    } else null,
                    isRead = json.optBoolean("isRead", false),
                    readAt = json.optString("readAt", null),
                    createdAt = json.optString("createdAt", null)
                )
                
                onMessageReceived(message)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to parse topic message", e)
            }
        }, { error ->
            Log.e(TAG, "Topic subscribe error", error)
        })
        
        disposable?.let { compositeDisposable.add(it) }
        return disposable
    }
    
    /**
     * Gửi tin nhắn qua WebSocket
     * Client gửi đến: /app/chat.send
     */
    fun sendMessage(chatId: Long?, receiverId: Long?, content: String) {
        if (!isConnected) {
            Log.e(TAG, "Cannot send message - WebSocket not connected")
            return
        }
        
        try {
            val payload = JSONObject()
            if (chatId != null) {
                payload.put("chatId", chatId)
            }
            if (receiverId != null) {
                payload.put("receiverId", receiverId)
            }
            payload.put("content", content)
            
            Log.d(TAG, "Sending message: $payload")
            stompClient?.send("/app/chat.send", payload.toString())?.subscribe({
                Log.d(TAG, "Message sent successfully")
            }, { error ->
                Log.e(TAG, "Failed to send message", error)
            })
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create message payload", e)
        }
    }
    
    /**
     * Ngắt kết nối WebSocket
     */
    fun disconnect() {
        try {
            compositeDisposable.clear()
            stompClient?.disconnect()
            isConnected = false
            Log.d(TAG, "WebSocket disconnected")
        } catch (e: Exception) {
            Log.e(TAG, "Error disconnecting WebSocket", e)
        }
    }
    
    fun isConnected(): Boolean = isConnected
}
