package com.example.Auto_BE.config;

import com.example.Auto_BE.utils.JwtUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.simp.config.ChannelRegistration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
import org.springframework.messaging.support.ChannelInterceptor;
import org.springframework.messaging.support.MessageHeaderAccessor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.util.Collections;
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
                .addInterceptors(new JwtHandshakeInterceptor(jwtUtils));
                // Không dùng .withSockJS() nữa - dùng WebSocket thuần
    }
    
    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        registration.interceptors(new ChannelInterceptor() {
            @Override
            public Message<?> preSend(Message<?> message, MessageChannel channel) {
                StompHeaderAccessor accessor = MessageHeaderAccessor.getAccessor(message, StompHeaderAccessor.class);
                
                if (accessor != null && StompCommand.SEND.equals(accessor.getCommand())) {
                    // Lấy username từ session attributes (đã lưu trong handshake)
                    String username = (String) accessor.getSessionAttributes().get("username");
                    
                    if (username != null) {
                        // Tạo authentication token và set vào SecurityContext
                        UsernamePasswordAuthenticationToken authentication = 
                            new UsernamePasswordAuthenticationToken(
                                username, 
                                null, 
                                Collections.singletonList(new SimpleGrantedAuthority("ROLE_USER"))
                            );
                        accessor.setUser(authentication);
                        
                        System.out.println("✅ Authenticated STOMP message from user: " + username);
                    } else {
                        System.out.println("⚠️ No username in session attributes for STOMP message");
                    }
                }
                
                return message;
            }
        });
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
            
            System.out.println("========== WebSocket Handshake Attempt ==========");
            System.out.println("Request URI: " + request.getURI());
            
            if (request instanceof ServletServerHttpRequest) {
                ServletServerHttpRequest servletRequest = (ServletServerHttpRequest) request;
                
                System.out.println("Request Method: " + servletRequest.getServletRequest().getMethod());
                System.out.println("Request Headers:");
                servletRequest.getServletRequest().getHeaderNames().asIterator()
                    .forEachRemaining(header -> 
                        System.out.println("  " + header + ": " + servletRequest.getServletRequest().getHeader(header))
                    );
                
                // Thử lấy token từ header trước
                String authHeader = servletRequest.getServletRequest().getHeader("Authorization");
                String token = null;
                
                if (authHeader != null && authHeader.startsWith("Bearer ")) {
                    token = authHeader.substring(7);
                    System.out.println("✅ Token from Authorization header: " + token.substring(0, Math.min(20, token.length())) + "...");
                }
                
                // Nếu không có token trong header, thử lấy từ query parameter
                if (token == null) {
                    String accessToken = servletRequest.getServletRequest().getParameter("access_token");
                    if (accessToken != null && !accessToken.isEmpty()) {
                        token = accessToken;
                        System.out.println("✅ Token from query parameter: " + token.substring(0, Math.min(20, token.length())) + "...");
                    } else {
                        System.out.println("❌ No access_token in query parameters");
                        System.out.println("Available parameters: " + servletRequest.getServletRequest().getParameterMap().keySet());
                    }
                }
                
                // Validate token nếu có
                if (token != null) {
                    try {
                        if (jwtUtils.validateToken(token)) {
                            String username = jwtUtils.getUsernameFromToken(token);
                            System.out.println("✅ Authentication successful for user: " + username);
                            
                            // Lưu username vào WebSocket session attributes
                            attributes.put("username", username);
                            System.out.println("========== Handshake SUCCESS ==========");
                            return true;
                        } else {
                            System.err.println("❌ Token validation failed");
                        }
                    } catch (Exception e) {
                        System.err.println("❌ JWT validation error: " + e.getMessage());
                        e.printStackTrace();
                        System.out.println("========== Handshake FAILED ==========");
                        return false;
                    }
                } else {
                    System.err.println("❌ No token provided in header or query parameter");
                }
            }
            
            System.out.println("========== Handshake FAILED ==========");
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
