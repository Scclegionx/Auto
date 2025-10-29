// package com.auto_fe.auto_fe.usecase

// import android.content.Context
// import android.util.Log
// import com.auto_fe.auto_fe.audio.VoiceManager
// import com.auto_fe.auto_fe.automation.calendar.CalendarAutomation
// import com.auto_fe.auto_fe.domain.VoiceEvent
// import com.auto_fe.auto_fe.domain.VoiceState
// import com.auto_fe.auto_fe.domain.VoiceStateMachine
// import com.auto_fe.auto_fe.core.CommandProcessor
// import java.util.Calendar

// /**
//  * State Machine cho luồng tạo sự kiện lịch tự động
//  * Xử lý toàn bộ flow từ nhận lệnh -> NLP parsing -> tạo sự kiện lịch
//  */
// class CalendarStateMachine(
//     private val context: Context,
//     private val voiceManager: VoiceManager,
//     private val calendarAutomation: CalendarAutomation
// ) : VoiceStateMachine() {
    
//     private val commandProcessor = CommandProcessor(context)

//     companion object {
//         private const val TAG = "CalendarStateMachine"
//     }

//     // Lưu context data trong quá trình xử lý
//     private var currentTitle: String = ""
//     private var currentLocation: String = ""
//     private var currentBegin: Long = 0L
//     private var currentEnd: Long = 0L
    
//     // Callback cho transcript
//     var onTranscriptUpdated: ((String) -> Unit)? = null

//     /**
//      * Định nghĩa State Transitions
//      */
//     override fun getNextState(currentState: VoiceState, event: VoiceEvent): VoiceState? {
//         return when (currentState) {

//             // IDLE -> Start listening
//             is VoiceState.Idle -> {
//                 when (event) {
//                     is VoiceEvent.StartRecording -> VoiceState.ListeningForCalendarCommand
//                     is VoiceEvent.UserCancelled -> VoiceState.Idle // Already idle
//                     else -> null
//                 }
//             }

//             // LISTENING -> Parse command
//             is VoiceState.ListeningForCalendarCommand -> {
//                 when (event) {
//                     is VoiceEvent.CalendarCommandReceived -> {
//                         VoiceState.ParsingCalendarCommand
//                     }
//                     is VoiceEvent.SpeechRecognitionFailed -> {
//                         VoiceState.Error("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ.")
//                     }
//                     is VoiceEvent.Timeout -> {
//                         VoiceState.Error("Tôi không nhận được phản hồi từ bạn.")
//                     }
//                     is VoiceEvent.UserCancelled -> {
//                         VoiceState.Idle
//                     }
//                     else -> null
//                 }
//             }

//             // PARSING -> Create calendar event
//             is VoiceState.ParsingCalendarCommand -> {
//                 when (event) {
//                     is VoiceEvent.CalendarCommandParsed -> {
//                         // Lưu data và chuyển sang tạo sự kiện
//                         currentTitle = event.title
//                         currentLocation = event.location
//                         currentBegin = event.begin
//                         currentEnd = event.end
//                         VoiceState.CreatingCalendarEvent(event.title, event.location, event.begin, event.end)
//                     }
//                     is VoiceEvent.CalendarCommandParseFailed -> {
//                         VoiceState.Error(event.reason)
//                     }
//                     is VoiceEvent.UserCancelled -> {
//                         VoiceState.Idle
//                     }
//                     else -> null
//                 }
//             }

//             // CREATING CALENDAR EVENT -> Handle result
//             is VoiceState.CreatingCalendarEvent -> {
//                 when (event) {
//                     is VoiceEvent.CalendarEventCreatedSuccessfully -> {
//                         VoiceState.Success
//                     }
//                     is VoiceEvent.CalendarEventCreationFailed -> {
//                         VoiceState.Error("Tạo sự kiện lịch thất bại: ${event.error}")
//                     }
//                     is VoiceEvent.UserCancelled -> {
//                         VoiceState.Idle
//                     }
//                     else -> null
//                 }
//             }

//             // Terminal states - no more transitions
//             is VoiceState.Success, is VoiceState.Error -> null

//             else -> null
//         }
//     }

//     /**
//      * Side Effects khi enter state
//      */
//     override fun onEnterState(state: VoiceState, event: VoiceEvent) {
//         Log.d(TAG, "onEnterState: ${state.getName()}")

//         when (state) {
//             is VoiceState.ListeningForCalendarCommand -> {
//                 // Reset VoiceManager trước khi bắt đầu
//                 voiceManager.resetBusyState()
//                 // Phát câu hỏi và bắt đầu lắng nghe
//                 speakAndListen("Bạn cần tôi trợ giúp điều gì?")
//             }

//             is VoiceState.ParsingCalendarCommand -> {
//                 // Parse command từ event
//                 if (event is VoiceEvent.CalendarCommandReceived) {
//                     parseCommandAsync(event.rawCommand)
//                 }
//             }

//             is VoiceState.CreatingCalendarEvent -> {
//                 // Tạo sự kiện lịch
//                 createCalendarEventAsync(state.title, state.location, state.begin, state.end)
//             }

//             is VoiceState.Success -> {
//                 speak("Đã tạo sự kiện lịch thành công.")
//             }

//             is VoiceState.Error -> {
//                 speak(state.errorMessage)
//             }

//             is VoiceState.Idle -> {
//                 // Reset state when cancelled
//                 if (event is VoiceEvent.UserCancelled) {
//                     speak("Đã hủy ghi âm")
//                     // Reset all context data
//                     currentTitle = ""
//                     currentLocation = ""
//                     currentBegin = 0L
//                     currentEnd = 0L
//                     // Reset VoiceManager busy state
//                     voiceManager.resetBusyState()
//                 }
//             }

//             else -> {
//                 // Do nothing
//             }
//         }
//     }

//     // ========== HELPER METHODS ==========

//     /**
//      * Phát giọng nói và bắt đầu lắng nghe
//      */
//     private fun speakAndListen(text: String, delaySeconds: Int = 2) {
//         // Sử dụng API mới với delay tùy chỉnh
//         voiceManager.textToSpeech(text, delaySeconds, object : VoiceManager.VoiceControllerCallback {
//             override fun onSpeechResult(spokenText: String) {
//                 handleSpeechResult(spokenText)
//             }

//             override fun onConfirmationResult(confirmed: Boolean) {}

//             override fun onError(error: String) {
//                 processEvent(VoiceEvent.SpeechRecognitionFailed)
//             }

//             override fun onAudioLevelChanged(level: Int) {
//                 // No audio level visualization for calendar creation
//             }
//         })
//     }

//     /**
//      * Chỉ phát giọng nói
//      */
//     private fun speak(text: String) {
//         voiceManager.speak(text)
//     }

//     /**
//      * Xử lý kết quả speech recognition dựa vào state hiện tại
//      */
//     private fun handleSpeechResult(spokenText: String) {
//         try {
//             Log.d(TAG, "Handling speech result: $spokenText")
            
//             // Update transcript in popup
//             onTranscriptUpdated?.invoke(spokenText)
            
//             when (currentState) {
//                 is VoiceState.ListeningForCalendarCommand -> {
//                     processEvent(VoiceEvent.CalendarCommandReceived(spokenText))
//                 }

//                 else -> {
//                     Log.w(TAG, "Unexpected speech result in state: ${currentState.getName()}")
//                 }
//             }
//         } catch (e: Exception) {
//             Log.e(TAG, "Error in handleSpeechResult: ${e.message}")
//             processEvent(VoiceEvent.SpeechRecognitionFailed)
//         }
//     }

//     /**
//      * Parse calendar command using NLP service - chỉ gửi raw command và nhận JSON response
//      */
//     private fun parseCommandAsync(command: String) {
//         Log.d(TAG, "Sending command to NLP: $command")
        
//         try {
//             // Add timeout protection
//             val timeoutHandler = android.os.Handler(android.os.Looper.getMainLooper())
//             var isCallbackExecuted = false
            
//             val timeoutRunnable = Runnable {
//                 if (!isCallbackExecuted) {
//                     Log.e(TAG, "NLP service timeout")
//                     isCallbackExecuted = true
//                     processEvent(VoiceEvent.CalendarCommandParseFailed("Lỗi hệ thống: Không nhận được phản hồi từ NLP service."))
//                 }
//             }
            
//             // Set timeout of 10 seconds
//             timeoutHandler.postDelayed(timeoutRunnable, 10000)
            
//             commandProcessor.processCommand(command, object : CommandProcessor.CommandProcessorCallback {
//                 override fun onCommandExecuted(success: Boolean, message: String) {
//                     if (!isCallbackExecuted) {
//                         isCallbackExecuted = true
//                         timeoutHandler.removeCallbacks(timeoutRunnable)
                        
//                         try {
//                             if (success) {
//                                 Log.d(TAG, "NLP processed successfully: $message")
//                                 // NLP đã xử lý xong và tạo sự kiện lịch, chuyển sang Success
//                                 processEvent(VoiceEvent.CalendarEventCreatedSuccessfully)
//                             } else {
//                                 Log.e(TAG, "NLP processing failed: $message")
//                                 processEvent(VoiceEvent.CalendarCommandParseFailed(message))
//                             }
//                         } catch (e: Exception) {
//                             Log.e(TAG, "Error in onCommandExecuted: ${e.message}")
//                             processEvent(VoiceEvent.CalendarCommandParseFailed("Lỗi xử lý lệnh: ${e.message}"))
//                         }
//                     }
//                 }

//                 override fun onError(error: String) {
//                     if (!isCallbackExecuted) {
//                         isCallbackExecuted = true
//                         timeoutHandler.removeCallbacks(timeoutRunnable)
                        
//                         try {
//                             Log.e(TAG, "NLP Error: $error")
//                             // Phân biệt lỗi hệ thống vs lỗi không hỗ trợ lệnh
//                             if (error.contains("Lỗi NLP") || error.contains("Server error") || error.contains("Connection") || 
//                                 error.contains("Lỗi kết nối") || error.contains("Lỗi không xác định")) {
//                                 processEvent(VoiceEvent.CalendarCommandParseFailed("Lỗi hệ thống: $error"))
//                             } else {
//                                 processEvent(VoiceEvent.CalendarCommandParseFailed("Hiện tại tôi không hỗ trợ lệnh này. Vui lòng vào tab 'Hướng dẫn' để xem các lệnh được hỗ trợ."))
//                             }
//                         } catch (e: Exception) {
//                             Log.e(TAG, "Error in onError: ${e.message}")
//                             processEvent(VoiceEvent.CalendarCommandParseFailed("Lỗi xử lý: ${e.message}"))
//                         }
//                     }
//                 }

//                 override fun onNeedConfirmation(command: String, receiver: String, message: String) {
//                     if (!isCallbackExecuted) {
//                         isCallbackExecuted = true
//                         timeoutHandler.removeCallbacks(timeoutRunnable)
                        
//                         try {
//                             Log.d(TAG, "NLP needs confirmation: $command -> title: $receiver, details: $message")
//                             // Parse calendar event từ NLP response
//                             val eventData = parseCalendarEventFromString(receiver, message)
//                             if (eventData != null) {
//                                 currentTitle = eventData.first
//                                 currentLocation = eventData.second
//                                 currentBegin = eventData.third
//                                 currentEnd = eventData.fourth
//                                 // Parse thành công, chuyển sang tạo sự kiện
//                                 processEvent(VoiceEvent.CalendarCommandParsed(currentTitle, currentLocation, currentBegin, currentEnd))
//                             } else {
//                                 processEvent(VoiceEvent.CalendarCommandParseFailed("Tôi không hiểu thông tin sự kiện. Vui lòng nói rõ hơn, ví dụ: 'Tạo sự kiện họp thứ 4 lúc 10 giờ'."))
//                             }
//                         } catch (e: Exception) {
//                             Log.e(TAG, "Error in onNeedConfirmation: ${e.message}")
//                             processEvent(VoiceEvent.CalendarCommandParseFailed("Lỗi xác nhận: ${e.message}"))
//                         }
//                     }
//                 }
//             })
//         } catch (e: Exception) {
//             Log.e(TAG, "Error in parseCommandAsync: ${e.message}")
//             processEvent(VoiceEvent.CalendarCommandParseFailed("Lỗi phân tích lệnh: ${e.message}"))
//         }
//     }

//     /**
//      * Parse calendar event từ string
//      */
//     private fun parseCalendarEventFromString(title: String, details: String): Quadruple<String, String, Long, Long>? {
//         try {
//             // Title từ receiver
//             val eventTitle = title.ifEmpty { "Sự kiện" }
            
//             // Location từ details hoặc mặc định
//             val location = if (details.isNotEmpty()) details else "Văn phòng"
            
//             // Tính thời gian: mặc định là thứ 4 tới lúc 10h sáng
//             val calendar = Calendar.getInstance()
//             calendar.set(Calendar.DAY_OF_WEEK, Calendar.WEDNESDAY)
//             calendar.set(Calendar.HOUR_OF_DAY, 10)
//             calendar.set(Calendar.MINUTE, 0)
//             calendar.set(Calendar.SECOND, 0)
//             calendar.set(Calendar.MILLISECOND, 0)
            
//             // Nếu thứ 4 đã qua trong tuần này, chuyển sang tuần sau
//             if (calendar.timeInMillis <= System.currentTimeMillis()) {
//                 calendar.add(Calendar.WEEK_OF_YEAR, 1)
//             }
            
//             val begin = calendar.timeInMillis
//             val end = calendar.timeInMillis + (60 * 60 * 1000) // 1 giờ sau
            
//             return Quadruple(eventTitle, location, begin, end)
//         } catch (e: Exception) {
//             Log.e(TAG, "Error parsing calendar event: ${e.message}")
//             return null
//         }
//     }

//     /**
//      * Tạo sự kiện lịch
//      */
//     private fun createCalendarEventAsync(title: String, location: String, begin: Long, end: Long) {
//         calendarAutomation.addEvent(title, location, begin, end, object : CalendarAutomation.CalendarCallback {
//             override fun onSuccess() {
//                 processEvent(VoiceEvent.CalendarEventCreatedSuccessfully)
//             }

//             override fun onError(error: String) {
//                 processEvent(VoiceEvent.CalendarEventCreationFailed(error))
//             }
//         })
//     }

//     /**
//      * Cleanup resources
//      */
//     fun cleanup() {
//         reset()
//         currentTitle = ""
//         currentLocation = ""
//         currentBegin = 0L
//         currentEnd = 0L
//         commandProcessor.release()
//     }
// }

// /**
//  * Data class để trả về 4 giá trị
//  */
// data class Quadruple<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)
