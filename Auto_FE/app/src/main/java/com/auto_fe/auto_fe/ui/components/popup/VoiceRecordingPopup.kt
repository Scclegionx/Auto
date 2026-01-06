package com.auto_fe.auto_fe.ui.components.popup

import android.content.Context
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.auto_fe.auto_fe.R
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

/**
 * Popup hiển thị khi ghi âm từ cửa sổ nổi
 * Có animation line và hiển thị transcript
 */
class VoiceRecordingPopup(private val context: Context) {
    
    private var windowManager: WindowManager? = null
    private var popupView: View? = null
    private var isShowing = false
    
    // UI components
    private var tvTimer: TextView? = null
    private var tvTranscript: TextView? = null
    private var animatedLine: View? = null
    private var btnCancel: Button? = null
    private var btnStop: Button? = null
    
    // Animation
    private val animationHandler = Handler(Looper.getMainLooper())
    private var animationRunnable: Runnable? = null
    private var startTime: Long = 0
    private var currentTranscript = ""
    
    // Callbacks
    var onCancelClick: (() -> Unit)? = null
    var onStopClick: (() -> Unit)? = null
    var onAudioLevelChanged: ((Int) -> Unit)? = null
    
    init {
        setupWindowManager()
        createPopupView()
    }
    
    private fun setupWindowManager() {
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }
    
    private fun createPopupView() {
        val inflater = LayoutInflater.from(context)
        popupView = inflater.inflate(R.layout.voice_recording_popup, null)
        
        // Initialize UI components
        tvTimer = popupView?.findViewById(R.id.tv_timer)
        tvTranscript = popupView?.findViewById(R.id.tv_transcript)
        animatedLine = popupView?.findViewById(R.id.animated_line)
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
    }
    
    fun show() {
        if (isShowing) return
        
        try {
            // Ensure popupView is not null
            if (popupView == null) {
                Log.e("VoiceRecordingPopup", "Popup view is null, recreating...")
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
            
            // Start timer and animation with delay
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                if (isShowing) {
                    startTimer()
                    startLineAnimation()
                }
            }, 200) // Delay 200ms to ensure view is properly laid out
            
            Log.d("VoiceRecordingPopup", "Popup shown")
        } catch (e: Exception) {
            Log.e("VoiceRecordingPopup", "Error showing popup: ${e.message}")
            isShowing = false
        }
    }
    
    fun hide() {
        if (!isShowing) return
        
        try {
            // Stop animations first
            stopTimer()
            stopLineAnimation()
            
            // Remove view from window manager
            if (popupView != null && windowManager != null) {
                windowManager?.removeView(popupView)
            }
            
            isShowing = false
            Log.d("VoiceRecordingPopup", "Popup hidden")
        } catch (e: Exception) {
            Log.e("VoiceRecordingPopup", "Error hiding popup: ${e.message}")
            isShowing = false // Ensure state is reset even if there's an error
        }
    }
    
    fun updateTranscript(text: String) {
        currentTranscript = text
        tvTranscript?.text = text
    }
    
    fun updateAudioLevel(level: Int) {
        try {
            // Scale level from 0-3 to 0-1 for animation
            val normalizedLevel = (level / 3f).coerceIn(0f, 1f)
            onAudioLevelChanged?.invoke(level)
            
            // Update line animation based on audio level (simplified)
            if (isShowing) {
                updateLineWidth(normalizedLevel)
            }
        } catch (e: Exception) {
            Log.e("VoiceRecordingPopup", "Error updating audio level: ${e.message}")
        }
    }
    
    private fun startTimer() {
        animationRunnable = object : Runnable {
            override fun run() {
                if (isShowing) {
                    val elapsed = System.currentTimeMillis() - startTime
                    val seconds = (elapsed / 1000).toInt()
                    val minutes = seconds / 60
                    val remainingSeconds = seconds % 60
                    
                    tvTimer?.text = String.format("%02d:%02d", minutes, remainingSeconds)
                    
                    animationHandler.postDelayed(this, 1000)
                }
            }
        }
        animationHandler.post(animationRunnable!!)
    }
    
    private fun stopTimer() {
        animationRunnable?.let { animationHandler.removeCallbacks(it) }
        animationRunnable = null
    }
    
    private fun startLineAnimation() {
        // Simplified animation - just show a static line
        try {
            animatedLine?.let { line ->
                if (line.isLaidOut) {
                    val parentWidth = (line.parent as? View)?.width ?: 0
                    if (parentWidth > 0) {
                        val targetWidth = (parentWidth * 0.5f).toInt().coerceAtLeast(10)
                        val layoutParams = line.layoutParams
                        if (layoutParams != null) {
                            layoutParams.width = targetWidth
                            line.layoutParams = layoutParams
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("VoiceRecordingPopup", "Error in startLineAnimation: ${e.message}")
        }
    }
    
    private fun stopLineAnimation() {
        // Stop line animation
        animationHandler.removeCallbacksAndMessages(null)
    }
    
    private fun updateLineWidth(level: Float) {
        try {
            animatedLine?.let { line ->
                // Check if view is laid out and still showing
                if (line.isLaidOut && isShowing) {
                    val parentWidth = (line.parent as? View)?.width ?: 0
                    if (parentWidth > 0) {
                        val targetWidth = (parentWidth * level).toInt().coerceAtLeast(10).coerceAtMost(parentWidth)
                        
                        // Use ViewGroup.LayoutParams instead of LinearLayout.LayoutParams
                        val layoutParams = line.layoutParams
                        if (layoutParams != null) {
                            layoutParams.width = targetWidth
                            line.layoutParams = layoutParams
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("VoiceRecordingPopup", "Error updating line width: ${e.message}")
        }
    }
    
    fun isVisible(): Boolean = isShowing
    
    fun release() {
        hide()
        onCancelClick = null
        onStopClick = null
        onAudioLevelChanged = null
    }
}
