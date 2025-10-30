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
import com.auto_fe.auto_fe.automation.msg.SMSAutomation
import com.auto_fe.auto_fe.automation.msg.SMSObserver
import com.auto_fe.auto_fe.core.CommandProcessor

class FloatingWindow(private val context: Context) {
    private var windowManager: WindowManager? = null
    private var floatingView: View? = null
    private var popupWindow: PopupWindow? = null
    private var audioManager: VoiceManager? = null
    private var smsAutomation: SMSAutomation? = null
    private var smsObserver: SMSObserver? = null

    private var commandProcessor: CommandProcessor? = null
    
    // Voice recording popup - Updated to use Liquid Wave version
    private var voiceRecordingPopup: LiquidWaveRecordingPopup? = null

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
        smsAutomation = SMSAutomation(context)
        smsObserver = SMSObserver(context)
        voiceRecordingPopup = LiquidWaveRecordingPopup(context)

        // Khởi tạo CommandProcessor
        initializeCommandProcessor()
    }

    /**
     * Khởi tạo CommandProcessor cho voice processing
     */
    private fun initializeCommandProcessor() {
        commandProcessor = CommandProcessor(context)

        // Setup callbacks cho CommandProcessor
        commandProcessor?.setupStateCallbacks(
            onSuccess = { message ->
                Log.d("FloatingWindow", "Command success: $message")
                Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                // Hide popup và reset UI
                voiceRecordingPopup?.hide()
                isRecording = false
                updateFloatingButtonIcon()
            },
            onError = { error ->
                Log.d("FloatingWindow", "Command error: $error")
                Toast.makeText(context, error, Toast.LENGTH_SHORT).show()
                // Hide popup và reset UI
                voiceRecordingPopup?.hide()
                isRecording = false
                updateFloatingButtonIcon()
            }
        )
        
        // Setup popup callbacks
        setupPopupCallbacks()
    }
    
    /**
     * Setup callbacks cho voice recording popup
     */
    private fun setupPopupCallbacks() {
        voiceRecordingPopup?.onStartClick = {
            Log.d("FloatingWindow", "Popup start recording clicked")
            // Bắt đầu ghi âm - tương tự như VoiceScreen
            startVoiceRecording()
        }
        
        voiceRecordingPopup?.onStopClick = {
            Log.d("FloatingWindow", "Popup stop recording clicked")
            // Dừng ghi âm
            stopVoiceRecording()
        }
    }

    /**
     * Bắt đầu ghi âm voice - GIỐNG HỆT VoiceScreen
     */
    private fun startVoiceRecording() {
        Log.d("FloatingWindow", "Starting voice recording")
        isRecording = true
        updateFloatingButtonIcon()
        
        // Hiển thị popup khi bắt đầu ghi âm
        voiceRecordingPopup?.show()
        
        // Chào hỏi và bắt đầu lắng nghe - GIỐNG HỆT VoiceScreen
        audioManager?.textToSpeech("Bạn cần tôi trợ giúp điều gì?", 0, object : VoiceManager.VoiceControllerCallback {
            override fun onSpeechResult(spokenText: String) {
                if (spokenText.isNotEmpty()) {
                    Log.d("FloatingWindow", "Speech result: $spokenText")
                    
                    // Gửi lệnh đến CommandProcessor - GIỐNG HỆT VoiceScreen
                    commandProcessor?.processCommand(spokenText, object : CommandProcessor.CommandProcessorCallback {
                        override fun onCommandExecuted(success: Boolean, message: String) {
                            Log.d("FloatingWindow", "Command executed: $success, $message")
                            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        }

                        override fun onError(error: String) {
                            Log.e("FloatingWindow", "Command error: $error")
                            Toast.makeText(context, error, Toast.LENGTH_SHORT).show()
                        }

                        override fun onNeedConfirmation(command: String, receiver: String, message: String) {
                            Log.d("FloatingWindow", "Confirmation handled by StateMachine: $command -> $receiver: $message")
                        }
                    })
                } else {
                    Log.w("FloatingWindow", "Empty speech result")
                    Toast.makeText(context, "Không nhận được lệnh", Toast.LENGTH_SHORT).show()
                    stopVoiceRecording()
                }
            }
            
            override fun onConfirmationResult(confirmed: Boolean) {
                // Not used in this context
            }

            override fun onError(error: String) {
                Log.e("FloatingWindow", "Speech recognition error: $error")
                Toast.makeText(context, "Lỗi nhận dạng giọng nói: $error", Toast.LENGTH_SHORT).show()
                stopVoiceRecording()
            }
            
            override fun onAudioLevelChanged(level: Int) {
                // Cập nhật voice level cho popup
                Log.d("FloatingWindow", "Audio level changed: $level")
                voiceRecordingPopup?.updateAudioLevel(level)
            }
        })
    }
    
    /**
     * Dừng ghi âm voice
     */
    private fun stopVoiceRecording() {
        Log.d("FloatingWindow", "Stopping voice recording")
        audioManager?.resetBusyState()
        isRecording = false
        updateFloatingButtonIcon()
        
        // Ẩn popup khi dừng ghi âm
        voiceRecordingPopup?.hide()
    }

    fun showFloatingWindow() {
        if (floatingView != null) {
            Log.d("FloatingWindow", "Floating window already exists")
            return
        }

        Log.d("FloatingWindow", "Creating floating window...")
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager

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

        smsObserver?.startObserving()
    }

    private fun createCircularFloatingButton(): View {
        val imageView = ImageView(context)
        Log.d("FloatingWindow", "Creating circular button")

        try {
            imageView.setImageResource(R.drawable.ic_mic_white)
            Log.d("FloatingWindow", "Icon set successfully")
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error setting icon: ${e.message}")
            imageView.setImageResource(android.R.drawable.ic_btn_speak_now)
        }

        imageView.setBackgroundResource(R.drawable.selector_circular_button)
        imageView.scaleType = ImageView.ScaleType.CENTER_INSIDE
        imageView.setPadding(16, 16, 16, 16)

        val size = (60 * context.resources.displayMetrics.density).toInt()
        imageView.layoutParams = ViewGroup.LayoutParams(size, size)

        // Click listener - Tương tự như VoiceScreen
        imageView.setOnClickListener {
            Log.d("FloatingWindow", "Click listener triggered - isRecording: $isRecording")

            if (!isRecording) {
                // Bắt đầu ghi âm
                startVoiceRecording()
            } else {
                // Dừng ghi âm
                stopVoiceRecording()
            }
        }

        // Touch listener cho drag functionality
        var isDragging = false

        imageView.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = layoutParams?.x ?: 0
                    initialY = layoutParams?.y ?: 0
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    isDragging = false
                    false
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaX = (event.rawX - initialTouchX).toInt()
                    val deltaY = (event.rawY - initialTouchY).toInt()

                    if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
                        isDragging = true
                        layoutParams?.x = initialX + deltaX
                        layoutParams?.y = initialY + deltaY
                        windowManager?.updateViewLayout(floatingView, layoutParams)
                        true
                    } else {
                        false
                    }
                }
                MotionEvent.ACTION_UP -> {
                    if (isDragging) {
                        snapToEdge()
                        true
                    } else {
                        false
                    }
                }
                else -> false
            }
        }

        return imageView
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

        layoutParams?.let { params ->
            val centerX = screenWidth / 2
            val newX = if (params.x < centerX) 0 else screenWidth - (floatingView?.width ?: 60)

            params.x = newX
            windowManager?.updateViewLayout(floatingView, params)
        }
    }

    private fun showCommandMenu() {
        if (popupWindow != null && popupWindow!!.isShowing) return

        val inflater = LayoutInflater.from(context)
        val menuView = inflater.inflate(R.layout.floating_menu, null)

        val recordButton = menuView.findViewById<Button>(R.id.btn_record)
        val closeButton = menuView.findViewById<Button>(R.id.btn_close)

        recordButton.setOnClickListener {
            hideCommandMenu()
            startVoiceRecording() // Sử dụng CommandProcessor
        }

        closeButton.setOnClickListener {
            hideCommandMenu()
        }

        popupWindow = PopupWindow(
            menuView,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            true
        )

        popupWindow?.isOutsideTouchable = true
        popupWindow?.isFocusable = true

        popupWindow?.setOnDismissListener {
            popupWindow = null
        }

        menuView.startAnimation(AnimationUtils.loadAnimation(context, R.anim.popup_enter))
        popupWindow?.showAsDropDown(floatingView)
    }

    private fun hideCommandMenu() {
        popupWindow?.let { popup ->
            if (popup.isShowing) {
                val menuView = popup.contentView
                menuView?.startAnimation(AnimationUtils.loadAnimation(context, R.anim.popup_exit))

                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    popup.dismiss()
                }, 150)
            }
        }
    }

    fun hideFloatingWindow() {
        smsObserver?.stopObserving()

        if (isRecording) {
            stopVoiceRecording()
        }

        // Hide and cleanup popup
        voiceRecordingPopup?.hide()
        voiceRecordingPopup = null

        floatingView?.let { view ->
            windowManager?.removeView(view)
        }
        floatingView = null
        windowManager = null
        audioManager?.release()
    }

    /**
     * Giải phóng tất cả resources
     */
    fun release() {
        try {
            // Cleanup CommandProcessor
            commandProcessor?.release()
            commandProcessor = null

            // Cleanup AudioManager
            audioManager?.release()
            audioManager = null

            // Cleanup SMSAutomation
            smsAutomation = null

            Log.d("FloatingWindow", "All resources released successfully")
        } catch (e: Exception) {
            Log.e("FloatingWindow", "Error releasing resources: ${e.message}")
        }
    }
}