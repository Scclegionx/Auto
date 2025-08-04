package com.auto_fe.services

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.auto_fe.widgets.FloatingWidget

class FloatingWidgetService : Service() {
    private lateinit var floatingWidget: FloatingWidget
    
    companion object {
        private const val TAG = "FloatingWidgetService"
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "FloatingWidgetService created")
        floatingWidget = FloatingWidget(this)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "FloatingWidgetService destroyed")
        floatingWidget.destroy()
    }
} 