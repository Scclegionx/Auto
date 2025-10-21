package com.auto_fe.auto_fe.ui.screens

import android.os.Build
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.AuthService
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.service.MyFirebaseMessagingService
import com.google.firebase.messaging.FirebaseMessaging
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

@Composable
fun AuthScreen(
    onLoginSuccess: (String) -> Unit = {} // Callback với accessToken
) {
    var isLoginMode by remember { mutableStateOf(true) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val authService = remember { AuthService() }

    // Login states
    var loginEmail by remember { mutableStateOf("") }
    var loginPassword by remember { mutableStateOf("") }
    var loginPasswordVisible by remember { mutableStateOf(false) }
    var isLoginLoading by remember { mutableStateOf(false) }

    // Register states
    var registerEmail by remember { mutableStateOf("") }
    var registerPassword by remember { mutableStateOf("") }
    var registerConfirmPassword by remember { mutableStateOf("") }
    var registerPasswordVisible by remember { mutableStateOf(false) }
    var registerConfirmPasswordVisible by remember { mutableStateOf(false) }
    var isRegisterLoading by remember { mutableStateOf(false) }
    var passwordError by remember { mutableStateOf<String?>(null) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        DarkGradientStart,
                        DarkGradientEnd
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Header
            Text(
                text = if (isLoginMode) "🔐 Đăng Nhập" else "📝 Đăng Ký",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = DarkPrimary
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = if (isLoginMode) "Chào mừng bạn quay trở lại!" else "Tạo tài khoản mới",
                fontSize = 16.sp,
                color = DarkOnSurface.copy(alpha = 0.7f),
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Content Card
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface.copy(alpha = 0.9f)
                ),
                shape = MaterialTheme.shapes.large,
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    if (isLoginMode) {
                        // LOGIN FORM
                        OutlinedTextField(
                            value = loginEmail,
                            onValueChange = { loginEmail = it },
                            label = { Text("Email", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.Email,
                                    contentDescription = "Email Icon",
                                    tint = DarkPrimary
                                )
                            },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = loginPassword,
                            onValueChange = { loginPassword = it },
                            label = { Text("Mật khẩu", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.Lock,
                                    contentDescription = "Password Icon",
                                    tint = DarkPrimary
                                )
                            },
                            trailingIcon = {
                                IconButton(onClick = { loginPasswordVisible = !loginPasswordVisible }) {
                                    Text(
                                        text = if (loginPasswordVisible) "👁️" else "🙈",
                                        fontSize = 20.sp
                                    )
                                }
                            },
                            visualTransformation = if (loginPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(24.dp))

                        Button(
                            onClick = {
                                if (loginEmail.isNotBlank() && loginPassword.isNotBlank()) {
                                    isLoginLoading = true
                                    scope.launch {
                                        try {
                                            val result = authService.login(loginEmail, loginPassword)
                                            result.fold(
                                                onSuccess = { response ->
                                                    val token = response.data?.accessToken
                                                    if (token != null) {
                                                        // Đăng ký device token sau khi login thành công
                                                        registerDeviceToken(
                                                            authService = authService,
                                                            token = token,
                                                            context = context,
                                                            scope = scope
                                                        )
                                                        
                                                        Toast.makeText(
                                                            context,
                                                            "✅ ${response.message}",
                                                            Toast.LENGTH_SHORT
                                                        ).show()
                                                        
                                                        // TODO: Save token to SharedPreferences
                                                        
                                                        // Chuyển sang màn hình danh sách đơn thuốc
                                                        onLoginSuccess(token)
                                                    }
                                                },
                                                onFailure = { error ->
                                                    Toast.makeText(
                                                        context,
                                                        "❌ ${error.message}",
                                                        Toast.LENGTH_LONG
                                                    ).show()
                                                }
                                            )
                                        } finally {
                                            isLoginLoading = false
                                        }
                                    }
                                }
                            },
                            enabled = !isLoginLoading && loginEmail.isNotBlank() && loginPassword.isNotBlank(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = DarkPrimary,
                                disabledContainerColor = DarkPrimary.copy(alpha = 0.5f)
                            ),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp)
                        ) {
                            if (isLoginLoading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(24.dp),
                                    color = DarkOnPrimary
                                )
                            } else {
                                Text(
                                    text = "Đăng Nhập",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = DarkOnPrimary
                                )
                            }
                        }
                    } else {
                        // REGISTER FORM
                        OutlinedTextField(
                            value = registerEmail,
                            onValueChange = { registerEmail = it },
                            label = { Text("Email", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.Email,
                                    contentDescription = "Email Icon",
                                    tint = DarkPrimary
                                )
                            },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = registerPassword,
                            onValueChange = {
                                registerPassword = it
                                passwordError = if (registerConfirmPassword.isNotEmpty() && it != registerConfirmPassword) {
                                    "Mật khẩu không khớp"
                                } else null
                            },
                            label = { Text("Mật khẩu", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.Lock,
                                    contentDescription = "Password Icon",
                                    tint = DarkPrimary
                                )
                            },
                            trailingIcon = {
                                IconButton(onClick = { registerPasswordVisible = !registerPasswordVisible }) {
                                    Text(
                                        text = if (registerPasswordVisible) "👁️" else "🙈",
                                        fontSize = 20.sp
                                    )
                                }
                            },
                            visualTransformation = if (registerPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            singleLine = true,
                            isError = passwordError != null,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                errorBorderColor = DarkError,
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        OutlinedTextField(
                            value = registerConfirmPassword,
                            onValueChange = {
                                registerConfirmPassword = it
                                passwordError = if (registerPassword.isNotEmpty() && it != registerPassword) {
                                    "Mật khẩu không khớp"
                                } else null
                            },
                            label = { Text("Xác nhận mật khẩu", color = DarkOnSurface.copy(alpha = 0.7f)) },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.Lock,
                                    contentDescription = "Confirm Password Icon",
                                    tint = DarkPrimary
                                )
                            },
                            trailingIcon = {
                                IconButton(onClick = { registerConfirmPasswordVisible = !registerConfirmPasswordVisible }) {
                                    Text(
                                        text = if (registerConfirmPasswordVisible) "👁️" else "🙈",
                                        fontSize = 20.sp
                                    )
                                }
                            },
                            visualTransformation = if (registerConfirmPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            singleLine = true,
                            isError = passwordError != null,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = DarkPrimary,
                                unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                                errorBorderColor = DarkError,
                                focusedTextColor = DarkOnSurface,
                                unfocusedTextColor = DarkOnSurface
                            ),
                            modifier = Modifier.fillMaxWidth()
                        )

                        if (passwordError != null) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = passwordError!!,
                                color = DarkError,
                                fontSize = 12.sp,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }

                        Spacer(modifier = Modifier.height(24.dp))

                        Button(
                            onClick = {
                                if (registerEmail.isNotBlank() && 
                                    registerPassword.isNotBlank() && 
                                    registerPassword == registerConfirmPassword) {
                                    isRegisterLoading = true
                                    scope.launch {
                                        try {
                                            val result = authService.register(registerEmail, registerPassword)
                                            result.fold(
                                                onSuccess = { response ->
                                                    Toast.makeText(
                                                        context,
                                                        "✅ ${response.message}\nVui lòng kiểm tra email để xác thực tài khoản!",
                                                        Toast.LENGTH_LONG
                                                    ).show()
                                                    // Switch to login mode
                                                    isLoginMode = true
                                                    registerEmail = ""
                                                    registerPassword = ""
                                                    registerConfirmPassword = ""
                                                },
                                                onFailure = { error ->
                                                    Toast.makeText(
                                                        context,
                                                        "❌ ${error.message}",
                                                        Toast.LENGTH_LONG
                                                    ).show()
                                                }
                                            )
                                        } finally {
                                            isRegisterLoading = false
                                        }
                                    }
                                }
                            },
                            enabled = !isRegisterLoading && 
                                     registerEmail.isNotBlank() && 
                                     registerPassword.isNotBlank() && 
                                     registerConfirmPassword.isNotBlank() &&
                                     registerPassword == registerConfirmPassword,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = DarkPrimary,
                                disabledContainerColor = DarkPrimary.copy(alpha = 0.5f)
                            ),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp)
                        ) {
                            if (isRegisterLoading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(24.dp),
                                    color = DarkOnPrimary
                                )
                            } else {
                                Text(
                                    text = "Đăng Ký",
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = DarkOnPrimary
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Toggle between Login/Register
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (isLoginMode) "Chưa có tài khoản? " else "Đã có tài khoản? ",
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            fontSize = 14.sp
                        )
                        TextButton(onClick = { 
                            isLoginMode = !isLoginMode
                            passwordError = null
                        }) {
                            Text(
                                text = if (isLoginMode) "Đăng ký ngay" else "Đăng nhập",
                                fontWeight = FontWeight.Bold,
                                color = DarkPrimary,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // API Info
            Text(
                text = "🔗 Kết nối với Auto_BE API",
                fontSize = 12.sp,
                color = DarkOnSurface.copy(alpha = 0.5f),
                textAlign = TextAlign.Center
            )
        }
    }
}

/**
 * Helper function để đăng ký device token sau khi login thành công
 */
private fun registerDeviceToken(
    authService: AuthService,
    token: String,
    context: android.content.Context,
    scope: kotlinx.coroutines.CoroutineScope
) {
    scope.launch {
        try {
            // Lấy thông tin device
            val deviceId = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ANDROID_ID
            )
            val deviceType = "Android"
            val deviceName = "${Build.MANUFACTURER} ${Build.MODEL}"
            
            // Lấy FCM token thật từ Firebase
            val fcmToken = try {
                FirebaseMessaging.getInstance().token.await()
            } catch (e: Exception) {
                Log.e("AuthScreen", "Failed to get FCM token from Firebase", e)
                // Fallback: Lấy token đã lưu hoặc dùng dummy
                MyFirebaseMessagingService.getSavedFCMToken(context) 
                    ?: "dummy_fcm_token_${System.currentTimeMillis()}"
            }
            
            Log.d("AuthScreen", "Registering device token...")
            Log.d("AuthScreen", "Device ID: $deviceId")
            Log.d("AuthScreen", "Device Type: $deviceType")
            Log.d("AuthScreen", "Device Name: $deviceName")
            Log.d("AuthScreen", "FCM Token: $fcmToken")
            
            val result = authService.registerDeviceToken(
                fcmToken = fcmToken,
                deviceId = deviceId,
                deviceType = deviceType,
                deviceName = deviceName,
                accessToken = token
            )
            
            result.fold(
                onSuccess = { response ->
                    Log.d("AuthScreen", "Device token registered: ${response.message}")
                    Toast.makeText(
                        context,
                        "📱 Device đã đăng ký: ${response.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                },
                onFailure = { error ->
                    Log.e("AuthScreen", "Failed to register device token: ${error.message}")
                    // Không hiển thị toast error để không làm phiền user
                }
            )
        } catch (e: Exception) {
            Log.e("AuthScreen", "Error registering device token", e)
        }
    }
}
