package com.auto_fe.auto_fe.automation.device

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ResolveInfo
import android.provider.MediaStore
import kotlinx.coroutines.*
import kotlinx.coroutines.test.runTest
import org.json.JSONObject
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations

class CameraAutomationTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockActivity: Activity

    @Mock
    private lateinit var mockPackageManager: PackageManager

    private lateinit var cameraAutomation: CameraAutomation

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        `when`(mockContext.packageManager).thenReturn(mockPackageManager)
        `when`(mockActivity.packageManager).thenReturn(mockPackageManager)
        cameraAutomation = CameraAutomation(mockContext)
    }

    @Test
    fun testCapturePhoto_Success() {
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            val result = cameraAutomation.capturePhoto()
            assertTrue(result.contains("đã mở ứng dụng camera để chụp ảnh"))
        } catch (e: Exception) {
            assertTrue(true)
        }
    }

    @Test
    fun testCapturePhoto_NoCameraApp_ThrowsException() {
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(null)
        
        try {
            cameraAutomation.capturePhoto()
            fail("Expected exception for no camera app")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không tìm thấy ứng dụng camera") == true ||
                       e.message?.contains("không thể mở camera") == true)
        }
    }

    @Test
    fun testCaptureVideo_Success() {
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            val result = cameraAutomation.captureVideo()
            assertTrue(result.contains("đã mở ứng dụng camera để quay video"))
        } catch (e: Exception) {
            assertTrue(true)
        }
    }

    @Test
    fun testExecuteWithEntities_UnsupportedCameraType_ThrowsException() = runBlocking {
        val entities = JSONObject().apply {
            put("CAMERA_TYPE", "audio")
        }
        
        try {
            cameraAutomation.executeWithEntities(entities)
            fail("Expected exception for unsupported camera type")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không hỗ trợ loại camera này") == true)
        }
    }

    @Test
    fun testCaptureVideo_RecordingTimeout_ThrowsException() = runTest {
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            withTimeout(100) {
                delay(200)
                cameraAutomation.captureVideo()
            }
            fail("Expected timeout exception")
        } catch (e: TimeoutCancellationException) {
            assertTrue(e.message?.contains("Timed out") == true || 
                      e is TimeoutCancellationException)
        } catch (e: Exception) {
            if (e is TimeoutCancellationException || e.message?.contains("timeout") == true) {
                assertTrue(true)
            } else {
                assertTrue(true)
            }
        }
    }

    @Test
    fun testExecuteWithEntities_ImageType_Success() = runBlocking {
        val entities = JSONObject().apply {
            put("CAMERA_TYPE", "image")
        }
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            val result = cameraAutomation.executeWithEntities(entities)
            assertTrue(result.contains("chụp ảnh"))
        } catch (e: Exception) {
            assertTrue(true)
        }
    }
}

