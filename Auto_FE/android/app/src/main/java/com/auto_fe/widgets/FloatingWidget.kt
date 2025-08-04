package com.auto_fe.widgets

import android.content.Context
import android.graphics.PixelFormat
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.Toast
import com.auto_fe.R
import com.auto_fe.audio.AudioRecorder
import com.auto_fe.core.AutomationCommand
import com.auto_fe.core.Constants

class FloatingWidget(private val context: Context) {
    private lateinit var windowManager: WindowManager
    private lateinit var floatingView: View
    private lateinit var widgetIcon: ImageView
    private lateinit var popupWindow: PopupWindow
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var automationCommand: AutomationCommand
    
    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f
    
    init {
        audioRecorder = AudioRecorder(context)
        automationCommand = AutomationCommand(context)
        setupFloatingWidget()
        setupPopupMenu()
    }
    
    private fun setupFloatingWidget() {
        // Tạo floating view với icon app
        floatingView = LayoutInflater.from(context).inflate(R.layout.floating_widget_icon, null)
        widgetIcon = floatingView.findViewById(R.id.widgetIcon)
        
        // Cấu hình window parameters
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        )
        
        params.gravity = Gravity.TOP or Gravity.START
        params.x = 0
        params.y = 100
        
        // Thêm floating view vào window
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        windowManager.addView(floatingView, params)
        
        // Xử lý touch events để di chuyển widget
        setupTouchListener(floatingView, params)
        
        // Xử lý click để hiển thị popup menu
        setupClickListener()
    }
    
    private fun setupTouchListener(view: View, params: WindowManager.LayoutParams) {
        var isDragging = false
        var dragThreshold = 10f // Ngưỡng để phân biệt drag và click
        
        view.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = params.x
                    initialY = params.y
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    isDragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaX = Math.abs(event.rawX - initialTouchX)
                    val deltaY = Math.abs(event.rawY - initialTouchY)
                    
                    if (deltaX > dragThreshold || deltaY > dragThreshold) {
                        isDragging = true
                        params.x = initialX + (event.rawX - initialTouchX).toInt()
                        params.y = initialY + (event.rawY - initialTouchY).toInt()
                        windowManager.updateViewLayout(floatingView, params)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!isDragging) {
                        // Nếu không drag thì mở popup menu
                        showPopupMenu()
                    }
                    true
                }
                else -> false
            }
        }
    }
    
    private fun setupClickListener() {
        // Không cần setup click listener riêng nữa vì đã xử lý trong touch listener
    }
    
    private fun setupPopupMenu() {
        val popupView = LayoutInflater.from(context).inflate(R.layout.widget_popup_menu, null)
        
        popupWindow = PopupWindow(
            popupView,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT
        ).apply {
            isFocusable = true
            isOutsideTouchable = true
            elevation = 10f
        }
        
        // Setup các nút trong popup menu
        setupPopupButtons(popupView)
    }
    
    private fun setupPopupButtons(popupView: View) {
        val recordButton = popupView.findViewById<View>(R.id.recordButton)
        val testMessageButton = popupView.findViewById<View>(R.id.testMessageButton)
        val settingsButton = popupView.findViewById<View>(R.id.settingsButton)
        val closeButton = popupView.findViewById<View>(R.id.closeButton)
        
        recordButton.setOnClickListener {
            toggleRecording()
            popupWindow.dismiss()
        }
        
        testMessageButton.setOnClickListener {
            testSendMessage()
            popupWindow.dismiss()
        }
        
        settingsButton.setOnClickListener {
            // TODO: Mở settings
            Toast.makeText(context, "Settings", Toast.LENGTH_SHORT).show()
            popupWindow.dismiss()
        }
        
        closeButton.setOnClickListener {
            popupWindow.dismiss()
        }
    }
    
    private fun showPopupMenu() {
        if (!popupWindow.isShowing) {
            popupWindow.showAsDropDown(widgetIcon)
        }
    }
    
    private fun toggleRecording() {
        if (audioRecorder.isRecording()) {
            audioRecorder.stopRecording()
            Toast.makeText(context, "Đã dừng ghi âm", Toast.LENGTH_SHORT).show()
        } else {
            audioRecorder.startRecording()
            Toast.makeText(context, "Bắt đầu ghi âm", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun testSendMessage() {
        // Test với dữ liệu fix cứng
        val command = Constants.COMMAND_SEND_MESSAGE
        val entities = Constants.TestData.TEST_ENTITIES
        val values = Constants.TestData.TEST_VALUES
        
        Toast.makeText(context, Constants.ToastMessages.STARTING_MESSAGE_TEST, Toast.LENGTH_SHORT).show()
        automationCommand.executeCommand(command, entities, values)
    }
    
    fun destroy() {
        if (::windowManager.isInitialized && ::floatingView.isInitialized) {
            windowManager.removeView(floatingView)
        }
        if (popupWindow.isShowing) {
            popupWindow.dismiss()
        }
        audioRecorder.release()
    }
} 