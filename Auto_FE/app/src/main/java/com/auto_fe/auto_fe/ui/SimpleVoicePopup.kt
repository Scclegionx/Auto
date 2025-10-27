package com.auto_fe.auto_fe.ui

import android.content.Context
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import com.auto_fe.auto_fe.R

/**
 * Simple voice recording popup without complex animations
 */
class SimpleVoicePopup(private val context: Context) {
    
    private var windowManager: WindowManager? = null
    private var popupView: View? = null
    private var isShowing = false
    
    // UI components
    private var tvTimer: TextView? = null
    private var tvTranscript: TextView? = null
    private var statusLine: View? = null
    private var btnCancel: Button? = null
    private var btnStop: Button? = null
    
    // Timer
    private val timerHandler = Handler(Looper.getMainLooper())
    private var timerRunnable: Runnable? = null
    private var startTime: Long = 0
    
    // Callbacks
    var onCancelClick: (() -> Unit)? = null
    var onStopClick: (() -> Unit)? = null
    
    init {
        setupWindowManager()
        createPopupView()
    }
    
    private fun setupWindowManager() {
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }
    
    private fun createPopupView() {
        try {
            val inflater = LayoutInflater.from(context)
            popupView = inflater.inflate(R.layout.simple_voice_popup, null)
            
            // Initialize UI components
            tvTimer = popupView?.findViewById(R.id.tv_timer)
            tvTranscript = popupView?.findViewById(R.id.tv_transcript)
            statusLine = popupView?.findViewById(R.id.status_line)
            btnCancel = popupView?.findViewById(R.id.btn_cancel)
            btnStop = popupView?.findViewById(R.id.btn_stop)
            
            // Setup click listeners
            btnCancel?.setOnClickListener {
                onCancelClick?.invoke()
                hide()
            }
            
            btnStop?.setOnClickListener {
                onStopClick?.invoke()
                hide()
            }
        } catch (e: Exception) {
            Log.e("SimpleVoicePopup", "Error creating popup view: ${e.message}")
        }
    }
    
    fun show() {
        if (isShowing) return
        
        try {
            // Ensure popupView is not null
            if (popupView == null) {
                Log.e("SimpleVoicePopup", "Popup view is null, recreating...")
                createPopupView()
            }
            
            val layoutParams = WindowManager.LayoutParams().apply {
                type = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                format = PixelFormat.TRANSLUCENT
                width = WindowManager.LayoutParams.WRAP_CONTENT
                height = WindowManager.LayoutParams.WRAP_CONTENT
                gravity = Gravity.CENTER
            }
            
            windowManager?.addView(popupView, layoutParams)
            isShowing = true
            startTime = System.currentTimeMillis()
            
            // Start timer
            startTimer()
            
            Log.d("SimpleVoicePopup", "Popup shown")
        } catch (e: Exception) {
            Log.e("SimpleVoicePopup", "Error showing popup: ${e.message}")
            isShowing = false
        }
    }
    
    fun hide() {
        if (!isShowing) return
        
        try {
            // Stop timer first
            stopTimer()
            
            // Remove view from window manager
            if (popupView != null && windowManager != null) {
                windowManager?.removeView(popupView)
            }
            
            isShowing = false
            Log.d("SimpleVoicePopup", "Popup hidden")
        } catch (e: Exception) {
            Log.e("SimpleVoicePopup", "Error hiding popup: ${e.message}")
            isShowing = false
        }
    }
    
    fun updateTranscript(text: String) {
        try {
            tvTranscript?.text = text
        } catch (e: Exception) {
            Log.e("SimpleVoicePopup", "Error updating transcript: ${e.message}")
        }
    }
    
    fun updateAudioLevel(level: Int) {
        // Simple visual feedback - just change the status line color
        try {
            statusLine?.alpha = (level / 3f).coerceIn(0.3f, 1f)
        } catch (e: Exception) {
            Log.e("SimpleVoicePopup", "Error updating audio level: ${e.message}")
        }
    }
    
    private fun startTimer() {
        timerRunnable = object : Runnable {
            override fun run() {
                if (isShowing) {
                    val elapsed = System.currentTimeMillis() - startTime
                    val seconds = (elapsed / 1000).toInt()
                    val minutes = seconds / 60
                    val remainingSeconds = seconds % 60
                    
                    tvTimer?.text = String.format("%02d:%02d", minutes, remainingSeconds)
                    
                    timerHandler.postDelayed(this, 1000)
                }
            }
        }
        timerHandler.post(timerRunnable!!)
    }
    
    private fun stopTimer() {
        timerRunnable?.let { timerHandler.removeCallbacks(it) }
        timerRunnable = null
    }
    
    fun isVisible(): Boolean = isShowing
    
    fun release() {
        hide()
        onCancelClick = null
        onStopClick = null
    }
}
