package com.example.Auto_BE.middleware;

import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.service.CustomUserDetailsService;
import com.example.Auto_BE.utils.JwtUtils;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import java.io.IOException;

import static com.example.Auto_BE.constants.ErrorMessages.UNAUTHENTICATED;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    private final JwtUtils jwtUtils;
    private final CustomUserDetailsService customUserDetailsService;

    public JwtAuthenticationFilter(JwtUtils jwtUtils, CustomUserDetailsService customUserDetailsService) {
        this.jwtUtils = jwtUtils;
        this.customUserDetailsService = customUserDetailsService;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getServletPath();
        return path.startsWith("/api/auth") ||
                path.startsWith("/swagger-ui") ||
                path.startsWith("/v3/api-docs") ||
                path.startsWith("/swagger-resources") ||
                path.startsWith("/webjars") ||
                path.equals("/swagger-ui.html") ||
                path.startsWith("/ws");
    }

    @Override
    protected void doFilterInternal(@NonNull HttpServletRequest request,
                                    @NonNull HttpServletResponse response,
                                    @NonNull FilterChain filterChain) throws IOException, ServletException {
        try {
            String authHeader = request.getHeader("Authorization");
            if (authHeader == null || !authHeader.startsWith("Bearer ")) {
                throw new BaseException.UnauthorizedException();
            }

            String jwtToken = authHeader.substring(7);

            if (!jwtUtils.validateAccessToken(jwtToken)) {
                throw new BaseException.UnauthorizedException();
            }
            String username = jwtUtils.getUsernameFromToken(jwtToken);
            UserDetails userDetails = customUserDetailsService.loadUserByUsername(username);

            UsernamePasswordAuthenticationToken authenticationToken =
                    new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());

            authenticationToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
            SecurityContextHolder.getContext().setAuthentication(authenticationToken);

        } catch (BaseException.UnauthorizedException e) {
            sendUnauthorizedResponse(response);
            return;
        }

        filterChain.doFilter(request, response);
    }

    private void sendUnauthorizedResponse(HttpServletResponse response) throws IOException {
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json");
        response.getWriter().write(UNAUTHENTICATED);
    }
}