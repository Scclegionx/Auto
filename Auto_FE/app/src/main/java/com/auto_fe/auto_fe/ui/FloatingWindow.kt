package com.auto_fe.auto_fe.ui

import android.content.Context
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.LayoutInflater
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.Toast
import android.util.Log
import android.app.AlertDialog
import android.view.animation.AnimationUtils
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import com.auto_fe.auto_fe.R
import com.auto_fe.auto_fe.audio.VoiceManager
import com.auto_fe.auto_fe.core.CommandProcessor
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.automation.msg.SMSObserver

class FloatingWindow(private val context: Context) {
    private var windowManager: WindowManager? = null
    private var floatingView: View? = null
    private var popupWindow: PopupWindow? = null
    private var audioManager: VoiceManager? = null
    private var commandProcessor: CommandProcessor? = null
    private var smsAutomation: SMSAutomation? = null
    private var smsObserver: SMSObserver? = null
    private var isWaitingForUserResponse = false
    private var pendingSMSMessage = ""
    
    // Drag functionality
    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f
    private var layoutParams: WindowManager.LayoutParams? = null
    
    // Recording state
    private var isRecording = false
    
    init {
        audioManager = VoiceManager.getInstance(context)
        commandProcessor = CommandProcessor(context)
        smsAutomation = SMSAutomation(context)
        smsObserver = SMSObserver(context)
        
        // SMSObserver tự động sử dụng singleton AudioManager
    }

    fun showFloatingWindow() {
        if (floatingView != null) {
            Log.d("FloatingWindow", "Floating window already exists")
            return
        }
        
        Log.d("FloatingWindow", "Creating floating window...")
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        
        // Tạo button tròn có thể kéo được
        floatingView = createCircularFloatingButton()
        
        layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        )
        
        layoutParams?.gravity = Gravity.TOP or Gravity.START
        layoutParams?.x = 0
        layoutParams?.y = 100
        
        try {
            windowManager?.addView(floatingView, layoutParams)
            Log.d("FloatingWindow", "Floating window added successfully")
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error adding floating window: ${e.message}")
        }
        
        // Bắt đầu lắng nghe SMS mới
        smsObserver?.startObserving()
    }
    
    private fun createCircularFloatingButton(): View {
        // Tạo ImageView tròn thay vì Button
        val imageView = ImageView(context)
        Log.d("FloatingWindow", "Creating circular button")
        
        // Thử set icon với fallback
        try {
            imageView.setImageResource(R.drawable.ic_mic_white)
            Log.d("FloatingWindow", "Icon set successfully")
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error setting icon: ${e.message}")
            // Fallback to default icon
            imageView.setImageResource(android.R.drawable.ic_btn_speak_now)
        }
        
        imageView.setBackgroundResource(R.drawable.selector_circular_button)
        imageView.scaleType = ImageView.ScaleType.CENTER_INSIDE
        imageView.setPadding(16, 16, 16, 16)
        
        // Thiết lập kích thước cố định cho button tròn
        val size = (60 * context.resources.displayMetrics.density).toInt()
        imageView.layoutParams = ViewGroup.LayoutParams(size, size)
        
        // Click listener riêng biệt
        imageView.setOnClickListener {
            Log.d("FloatingWindow", "Click listener triggered - isRecording: $isRecording")
            if (isRecording) {
                stopRecording()
            } else {
                startDirectRecording()
            }
        }
        
        // Touch listener cho drag functionality
        var isDragging = false
        var startTime = 0L
        
        imageView.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = layoutParams?.x ?: 0
                    initialY = layoutParams?.y ?: 0
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    startTime = System.currentTimeMillis()
                    isDragging = false
                    false // Không consume event để click listener hoạt động
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaX = (event.rawX - initialTouchX).toInt()
                    val deltaY = (event.rawY - initialTouchY).toInt()
                    
                    // Chỉ drag nếu di chuyển đủ xa
                    if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
                        isDragging = true
                        layoutParams?.x = initialX + deltaX
                        layoutParams?.y = initialY + deltaY
                        windowManager?.updateViewLayout(floatingView, layoutParams)
                        true // Consume event khi drag
                    } else {
                        false // Không consume event khi không drag
                    }
                }
                MotionEvent.ACTION_UP -> {
                    if (isDragging) {
                        // Drag - snap to edges
                        Log.d("FloatingWindow", "Drag detected")
                        snapToEdge()
                        true
                    } else {
                        // Không drag - để click listener xử lý
                        false
                    }
                }
                else -> false
            }
        }
        
        return imageView
    }
    
    private fun startDirectRecording() {
        Log.d("FloatingWindow", "startDirectRecording called")
        if (isRecording) {
            Log.d("FloatingWindow", "Already recording, returning")
            return
        }
        
        Log.d("FloatingWindow", "Starting recording...")
        isRecording = true
        updateFloatingButtonIcon()
        
        // Bắt đầu ghi âm
        audioManager?.startVoiceInteraction(object : VoiceManager.AudioManagerCallback {
            override fun onSpeechResult(spokenText: String) {
                isRecording = false
                updateFloatingButtonIcon()
                
                // Xử lý lệnh
                commandProcessor?.processCommand(spokenText, object : CommandProcessor.CommandProcessorCallback {
                    override fun onCommandExecuted(success: Boolean, message: String) {
                        audioManager?.speak(message)
                        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                    }
                    override fun onError(error: String) {
                        if (error.contains("danh bạ có tên gần giống")) {
                            isWaitingForUserResponse = true
                            pendingSMSMessage = extractMessageFromCommand(spokenText)
                            audioManager?.speak(error)
                        } else {
                            audioManager?.speak(error)
                        }
                        Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                    }
                })
            }
            override fun onConfirmationResult(confirmed: Boolean) {}
            override fun onError(error: String) {
                isRecording = false
                updateFloatingButtonIcon()
                audioManager?.speak(error)
                Toast.makeText(context, error, Toast.LENGTH_LONG).show()
            }
        })
    }
    
    private fun stopRecording() {
        if (!isRecording) return
        
        isRecording = false
        updateFloatingButtonIcon()
        
        // Không cần release audioManager ở đây vì nó là singleton
    }
    
    private fun updateFloatingButtonIcon() {
        val imageView = floatingView as? ImageView
        Log.d("FloatingWindow", "updateFloatingButtonIcon - isRecording: $isRecording")
        if (imageView != null) {
            try {
                val iconRes = if (isRecording) R.drawable.ic_stop_white else R.drawable.ic_mic_white
                imageView.setImageResource(iconRes)
                Log.d("FloatingWindow", "Icon updated successfully")
            } catch (e: Exception) {
                Log.e("FloatingWindow", "Error updating icon: ${e.message}")
            }
        } else {
            Log.e("FloatingWindow", "floatingView is not ImageView")
        }
    }
    
    
    private fun snapToEdge() {
        val displayMetrics = context.resources.displayMetrics
        val screenWidth = displayMetrics.widthPixels
        val screenHeight = displayMetrics.heightPixels
        
        layoutParams?.let { params ->
            val centerX = screenWidth / 2
            val newX = if (params.x < centerX) 0 else screenWidth - (floatingView?.width ?: 60)
            
            params.x = newX
            windowManager?.updateViewLayout(floatingView, params)
        }
    }

    private fun showCommandMenu() {
        // Sửa bug: Kiểm tra popupWindow đã dismiss chưa thay vì null
        if (popupWindow != null && popupWindow!!.isShowing) return
        
        val inflater = LayoutInflater.from(context)
        val menuView = inflater.inflate(R.layout.floating_menu, null)

        val recordButton = menuView.findViewById<Button>(R.id.btn_record)
        val testSmsButton = menuView.findViewById<Button>(R.id.btn_test_sms)
        val testNlpButton = menuView.findViewById<Button>(R.id.btn_test_nlp)
        val closeButton = menuView.findViewById<Button>(R.id.btn_close)

        recordButton.setOnClickListener {
            hideCommandMenu()
            startAudioRecording()
        }

        testSmsButton.setOnClickListener {
            hideCommandMenu()
            testSMSFunction()
        }

        testNlpButton.setOnClickListener {
            hideCommandMenu()
            testNLPFlow()
        }

        closeButton.setOnClickListener {
            hideCommandMenu()
        }

        // Tạo popup window với animation
        popupWindow = PopupWindow(
            menuView,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            true
        )
        
        // Tự động đóng khi click outside
        popupWindow?.isOutsideTouchable = true
        popupWindow?.isFocusable = true
        
        // Thêm dismiss listener để reset popupWindow về null
        popupWindow?.setOnDismissListener {
            popupWindow = null
        }
        
        // Thêm animation tùy chỉnh
        menuView.startAnimation(AnimationUtils.loadAnimation(context, com.auto_fe.auto_fe.R.anim.popup_enter))

        popupWindow?.showAsDropDown(floatingView)
    }

    private fun hideCommandMenu() {
        popupWindow?.let { popup ->
            if (popup.isShowing) {
            // Thêm animation khi đóng
            val menuView = popup.contentView
            menuView?.startAnimation(AnimationUtils.loadAnimation(context, com.auto_fe.auto_fe.R.anim.popup_exit))

            // Đóng popup sau khi animation hoàn thành
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                popup.dismiss()
            }, 150)
        }
        }
        // Không cần set popupWindow = null ở đây vì đã có setOnDismissListener
    }

    private fun startAudioRecording() {
        if (isWaitingForUserResponse) {
            // Đang chờ phản hồi từ người dùng về danh bạ
            handleUserResponseForSMS()
        } else {
            // Luồng bình thường - ghi âm lệnh mới
            audioManager?.startVoiceInteraction(object : VoiceManager.AudioManagerCallback {
                override fun onSpeechResult(spokenText: String) {
                    audioManager?.confirmCommand(spokenText, object : VoiceManager.AudioManagerCallback {
                        override fun onSpeechResult(spokenText: String) {}
                        override fun onConfirmationResult(confirmed: Boolean) {
                            if (confirmed) {
                                commandProcessor?.processCommand(spokenText, object : CommandProcessor.CommandProcessorCallback {
                                    override fun onCommandExecuted(success: Boolean, message: String) {
                                        // Thông báo thành công bằng giọng nói
                                        audioManager?.speak(message)
                                        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
                                    }
                                    override fun onError(error: String) {
                                        // Kiểm tra xem có phải lỗi cần xác nhận danh bạ không
                                        if (error.contains("danh bạ có tên gần giống")) {
                                            // Lưu thông tin để xử lý hội thoại
                                            isWaitingForUserResponse = true
                                            pendingSMSMessage = extractMessageFromCommand(spokenText)
                                            // Thông báo bằng giọng nói
                                            audioManager?.speak(error)
                                        } else {
                                            // Thông báo lỗi bình thường
                                            audioManager?.speak(error)
                                        }
                                        Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                                    }
                                })
                            } else {
                                audioManager?.speak("Lệnh đã bị hủy")
                                Toast.makeText(context, "Lệnh đã bị hủy", Toast.LENGTH_SHORT).show()
                            }
                        }
                        override fun onError(error: String) {
                            audioManager?.speak(error)
                            Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                        }
                    })
                }
                override fun onConfirmationResult(confirmed: Boolean) {}
                override fun onError(error: String) {
                    audioManager?.speak(error)
                    Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                }
            })
        }
    }
    
    /**
     * Xử lý phản hồi từ người dùng khi cần xác nhận danh bạ
     */
    private fun handleUserResponseForSMS() {
        audioManager?.startVoiceInteraction(object : VoiceManager.AudioManagerCallback {
            override fun onSpeechResult(spokenText: String) {
                // Xử lý phản hồi của người dùng
                smsAutomation?.handleUserResponse(spokenText, pendingSMSMessage, object : SMSAutomation.SMSConversationCallback {
                    override fun onSuccess() {
                        isWaitingForUserResponse = false
                        pendingSMSMessage = ""
                        audioManager?.speak("Đã gửi tin nhắn thành công")
                        Toast.makeText(context, "Đã gửi tin nhắn thành công", Toast.LENGTH_LONG).show()
                    }
                    override fun onError(error: String) {
                        isWaitingForUserResponse = false
                        pendingSMSMessage = ""
                        audioManager?.speak(error)
                        Toast.makeText(context, error, Toast.LENGTH_LONG).show()
                    }
                    override fun onNeedConfirmation(similarContacts: List<String>, originalName: String) {
                        // Không cần xử lý ở đây vì đã xử lý trong luồng chính
                    }
                })
            }
            override fun onConfirmationResult(confirmed: Boolean) {}
            override fun onError(error: String) {
                audioManager?.speak(error)
                Toast.makeText(context, error, Toast.LENGTH_LONG).show()
            }
        })
    }
    
    /**
     * Trích xuất nội dung tin nhắn từ lệnh gốc
     */
    private fun extractMessageFromCommand(command: String): String {
        // Tạm thời trả về tin nhắn mặc định, có thể cải thiện sau
        return "Tin nhắn từ Auto FE"
    }
    
    private fun testSMSFunction() {
        Log.d("FloatingWindow", "Starting SMS test...")
        val smsAutomation = SMSAutomation(context)
        smsAutomation.sendSMS("mom", "con sắp về", object : SMSAutomation.SMSCallback {
            override fun onSuccess() {
                Log.d("FloatingWindow", "SMS test successful!")
                showTestResult("Test SMS thành công!")
            }
            override fun onError(error: String) {
                Log.e("FloatingWindow", "SMS test error: $error")
                showTestResult("Test SMS lỗi: $error")
            }
        })
    }

    private fun testNLPFlow() {
        Log.d("FloatingWindow", "Starting NLP flow test...")
        val testCommand = "nhắn tin cho mom là con sắp về"
        Log.d("FloatingWindow", "Testing command: $testCommand")

        commandProcessor?.processCommand(testCommand, object : CommandProcessor.CommandProcessorCallback {
            override fun onCommandExecuted(success: Boolean, message: String) {
                Log.d("FloatingWindow", "NLP flow test result: $success - $message")
                showTestResult("NLP Flow Test: $message")
            }
            override fun onError(error: String) {
                Log.e("FloatingWindow", "NLP flow test error: $error")
                showTestResult("NLP Flow Test Error: $error")
            }
        })
    }

    private fun showTestResult(message: String) {
        try {
            val builder = AlertDialog.Builder(context)
            builder.setTitle("Test Result")
            builder.setMessage(message)
            builder.setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            builder.show()
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error showing dialog: ${e.message}")
        }
    }

    fun hideFloatingWindow() {
        // Dừng lắng nghe SMS
        smsObserver?.stopObserving()
        
        // Dừng ghi âm nếu đang ghi
        if (isRecording) {
            stopRecording()
        }
        
        floatingView?.let { view ->
            windowManager?.removeView(view)
        }
        floatingView = null
        windowManager = null
        audioManager?.release()
    }

    /**
     * Giải phóng tất cả resources để tránh memory leak
     */
    fun release() {
        try {
            // Cleanup CommandProcessor (bao gồm PhoneAutomation TTS)
            commandProcessor?.release()
            commandProcessor = null

            // Cleanup AudioManager
            audioManager?.release()
            audioManager = null

            Log.d("FloatingWindow", "All resources released successfully")
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error releasing resources: ${e.message}")
        }
    }
}