package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.service.CloudinaryService;
import com.example.Auto_BE.utils.JwtUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;

/**
 * Controller để upload file/ảnh lên Cloudinary
 */
@RestController
@RequestMapping("/api/upload")
@RequiredArgsConstructor
public class FileUploadController {
    
    private final CloudinaryService cloudinaryService;
    private final JwtUtils jwtUtils;
    
    /**
     * Upload ảnh lên Cloudinary
     */
    @PostMapping("/image")
    public ResponseEntity<BaseResponse<Map<String, Object>>> uploadImage(
            @RequestParam("file") MultipartFile file,
            @RequestHeader("Authorization") String authHeader) {
        
        long startTime = System.currentTimeMillis();
        System.out.println("[UPLOAD-IMAGE] START - File: " + file.getOriginalFilename() + ", Size: " + file.getSize() + " bytes");
        
        try {
            // Verify token
            String token = authHeader.substring(7);
            Long userId = jwtUtils.getUserIdFromToken(token);
            System.out.println("[UPLOAD-IMAGE] User ID: " + userId);
            
            // Upload to Cloudinary
            System.out.println("[UPLOAD-IMAGE] Starting Cloudinary upload...");
            Map<String, Object> uploadResult = cloudinaryService.upload(file);
            System.out.println("[UPLOAD-IMAGE] Cloudinary upload completed in " + (System.currentTimeMillis() - startTime) + "ms");
            
            Map<String, Object> response = new HashMap<>();
            response.put("url", uploadResult.get("url"));
            response.put("publicId", uploadResult.get("public_id"));
            response.put("format", uploadResult.get("format"));
            response.put("resourceType", uploadResult.get("resource_type"));
            response.put("bytes", uploadResult.get("bytes"));
            
            System.out.println("[UPLOAD-IMAGE] SUCCESS - Total time: " + (System.currentTimeMillis() - startTime) + "ms, URL: " + response.get("url"));
            
            return ResponseEntity.ok(BaseResponse.<Map<String, Object>>builder()
                    .status("success")
                    .message("Image uploaded successfully")
                    .data(response)
                    .build());
                    
        } catch (Exception e) {
            System.out.println("[UPLOAD-IMAGE] ERROR after " + (System.currentTimeMillis() - startTime) + "ms: " + e.getMessage());
            return ResponseEntity.badRequest().body(BaseResponse.<Map<String, Object>>builder()
                    .status("error")
                    .message("Failed to upload image: " + e.getMessage())
                    .build());
        }
    }
    
    /**
     * Upload file (non-image) lên Cloudinary
     */
    @PostMapping("/file")
    public ResponseEntity<BaseResponse<Map<String, Object>>> uploadFile(
            @RequestParam("file") MultipartFile file,
            @RequestHeader("Authorization") String authHeader) {
        
        long startTime = System.currentTimeMillis();
        System.out.println("[UPLOAD-FILE] START - File: " + file.getOriginalFilename() + ", Size: " + file.getSize() + " bytes");
        
        try {
            // Verify token
            String token = authHeader.substring(7);
            Long userId = jwtUtils.getUserIdFromToken(token);
            System.out.println("[UPLOAD-FILE] User ID: " + userId);
            
            // Upload to Cloudinary as raw file
            System.out.println("[UPLOAD-FILE] Starting Cloudinary upload...");
            Map<String, Object> uploadResult = cloudinaryService.upload(file);
            System.out.println("[UPLOAD-FILE] Cloudinary upload completed in " + (System.currentTimeMillis() - startTime) + "ms");
            
            Map<String, Object> response = new HashMap<>();
            response.put("url", uploadResult.get("url"));
            response.put("publicId", uploadResult.get("public_id"));
            response.put("format", uploadResult.get("format"));
            response.put("resourceType", uploadResult.get("resource_type"));
            response.put("bytes", uploadResult.get("bytes"));
            response.put("originalFilename", file.getOriginalFilename());
            
            System.out.println("[UPLOAD-FILE] SUCCESS - Total time: " + (System.currentTimeMillis() - startTime) + "ms, URL: " + response.get("url"));
            
            return ResponseEntity.ok(BaseResponse.<Map<String, Object>>builder()
                    .status("success")
                    .message("File uploaded successfully")
                    .data(response)
                    .build());
                    
        } catch (Exception e) {
            System.out.println("[UPLOAD-FILE] ERROR after " + (System.currentTimeMillis() - startTime) + "ms: " + e.getMessage());
            return ResponseEntity.badRequest().body(BaseResponse.<Map<String, Object>>builder()
                    .status("error")
                    .message("Failed to upload file: " + e.getMessage())
                    .build());
        }
    }
}
