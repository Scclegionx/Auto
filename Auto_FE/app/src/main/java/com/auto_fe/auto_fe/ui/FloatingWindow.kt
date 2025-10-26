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
import com.auto_fe.auto_fe.usecase.SendSMSStateMachine
import com.auto_fe.auto_fe.domain.VoiceState
import com.auto_fe.auto_fe.domain.VoiceEvent

/**
 * FloatingWindow - Refactored với State Machine
 *
 * THAY ĐỔI CHÍNH:
 * - Loại bỏ các biến state thủ công (isWaitingForUserResponse, pendingSMSMessage, etc.)
 * - Sử dụng SendSMSStateMachine để quản lý state
 * - Logic đơn giản hóa, dễ maintain hơn
 */
class FloatingWindow(private val context: Context) {
    private var windowManager: WindowManager? = null
    private var floatingView: View? = null
    private var popupWindow: PopupWindow? = null
    private var audioManager: VoiceManager? = null
    private var smsAutomation: SMSAutomation? = null
    private var smsObserver: SMSObserver? = null

    // State Machine thay thế cho các biến state cũ
    private var smsStateMachine: SendSMSStateMachine? = null

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

        // Khởi tạo State Machine
        initializeStateMachine()
    }

    /**
     * Khởi tạo State Machine cho SMS flow
     */
    private fun initializeStateMachine() {
        smsStateMachine = SendSMSStateMachine(
            context = context,
            voiceManager = audioManager!!,
            smsAutomation = smsAutomation!!
        )

        // Lắng nghe state changes
        smsStateMachine?.onStateChanged = { oldState, newState ->
            Log.d("FloatingWindow", "State changed: ${oldState.getName()} -> ${newState.getName()}")

            // Update UI based on state
            handleStateChange(newState)

            // Show toast for debugging (có thể bỏ ở production)
            if (newState is VoiceState.Error || newState is VoiceState.Success) {
                val message = when (newState) {
                    is VoiceState.Success -> "Thành công!"
                    is VoiceState.Error -> newState.errorMessage
                    else -> ""
                }
                Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
            }
        }

        // Lắng nghe events (for debugging)
        smsStateMachine?.onEventProcessed = { event ->
            Log.d("FloatingWindow", "Event processed: ${event.getName()}")
        }
    }

    /**
     * Xử lý UI changes khi state thay đổi
     */
    private fun handleStateChange(newState: VoiceState) {
        when (newState) {
            is VoiceState.ListeningForSMSCommand -> {
                // Update button to recording state
                isRecording = true
                updateFloatingButtonIcon()
            }

            is VoiceState.Success, is VoiceState.Error -> {
                // Reset về idle state sau 2 giây
                android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                    smsStateMachine?.reset()
                    isRecording = false
                    updateFloatingButtonIcon()
                }, 2000)
            }

            else -> {
                // Keep recording state for other states
                isRecording = true
                updateFloatingButtonIcon()
            }
        }
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

        // Click listener - Disable khi đang xử lý
        imageView.setOnClickListener {
            Log.d("FloatingWindow", "Click listener triggered - current state: ${smsStateMachine?.getCurrentStateName()}")

            // Chỉ cho phép bấm khi ở Idle hoặc Terminal state
            if (smsStateMachine?.isTerminal() == true ||
                smsStateMachine?.currentState is VoiceState.Idle) {
                // Bắt đầu SMS flow mới
                startSMSFlow()
            } else {
                // Đang trong flow - Hủy flow hiện tại
                Log.d("FloatingWindow", "Cancelling current flow")
                smsStateMachine?.processEvent(VoiceEvent.UserCancelled)
                Toast.makeText(context, "Đã hủy ghi âm", Toast.LENGTH_SHORT).show()
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

    /**
     * Bắt đầu SMS flow - GỌI STATE MACHINE
     */
    private fun startSMSFlow() {
        Log.d("FloatingWindow", "Starting SMS flow")

        // Reset state machine nếu cần
        if (smsStateMachine?.isTerminal() == true) {
            smsStateMachine?.reset()
        }

        // Trigger StartRecording event
        smsStateMachine?.processEvent(VoiceEvent.StartRecording)
    }

    /**
     * Dừng flow hiện tại
     */
    private fun stopCurrentFlow() {
        Log.d("FloatingWindow", "Stopping current flow")

        // Cancel current flow
        smsStateMachine?.processEvent(VoiceEvent.UserCancelled)

        // Reset UI
        isRecording = false
        updateFloatingButtonIcon()
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
            startSMSFlow() // Sử dụng State Machine
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
            stopCurrentFlow()
        }

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
            // Cleanup State Machine
            smsStateMachine?.cleanup()
            smsStateMachine = null

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