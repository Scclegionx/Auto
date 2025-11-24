package com.example.Auto_BE.config;

import com.example.Auto_BE.utils.JwtUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.util.Map;

/**
 * WebSocket configuration cho chat real-time
 */
@Configuration
@EnableWebSocketMessageBroker
@RequiredArgsConstructor
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    
    private final JwtUtils jwtUtils;
    
    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // Enable simple memory-based message broker
        config.enableSimpleBroker("/topic", "/queue");
        
        // Prefix cho messages từ client
        config.setApplicationDestinationPrefixes("/app");
        
        // Prefix cho user-specific messages
        config.setUserDestinationPrefix("/user");
    }
    
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws/chat")
                .setAllowedOriginPatterns("*")
                .addInterceptors(new JwtHandshakeInterceptor(jwtUtils))
                .withSockJS();
    }
    
    /**
     * Interceptor để xác thực JWT token khi handshake WebSocket
     */
    @RequiredArgsConstructor
    private static class JwtHandshakeInterceptor implements HandshakeInterceptor {
        
        private final JwtUtils jwtUtils;
        
        @Override
        public boolean beforeHandshake(ServerHttpRequest request, 
                                        ServerHttpResponse response,
                                        WebSocketHandler wsHandler, 
                                        Map<String, Object> attributes) throws Exception {
            
            if (request instanceof ServletServerHttpRequest) {
                ServletServerHttpRequest servletRequest = (ServletServerHttpRequest) request;
                String authHeader = servletRequest.getServletRequest().getHeader("Authorization");
                
                if (authHeader != null && authHeader.startsWith("Bearer ")) {
                    String token = authHeader.substring(7);
                    
                    try {
                        // Validate token
                        if (jwtUtils.validateToken(token)) {
                            String username = jwtUtils.getUsernameFromToken(token);
                            
                            // Lưu username vào WebSocket session attributes
                            attributes.put("username", username);
                            return true;
                        }
                    } catch (Exception e) {
                        System.err.println("JWT validation failed: " + e.getMessage());
                        return false;
                    }
                }
            }
            
            return false; // Không có token hoặc token không hợp lệ
        }
        
        @Override
        public void afterHandshake(ServerHttpRequest request, 
                                    ServerHttpResponse response,
                                    WebSocketHandler wsHandler, 
                                    Exception exception) {
            // Do nothing
        }
    }
}
