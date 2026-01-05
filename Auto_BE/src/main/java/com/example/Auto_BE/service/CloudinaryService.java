package com.example.Auto_BE.service;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

@Service
@Slf4j
@RequiredArgsConstructor
public class CloudinaryService {

    private final Cloudinary cloudinary;

    /**
     * Upload file (image/document) lên Cloudinary
     * @param file File cần upload
     * @return Map chứa thông tin file đã upload (url, public_id, format, resource_type, bytes)
     */
    public Map<String, Object> upload(MultipartFile file) throws IOException {
        try {
            // Generate unique filename
            String publicId = "chat_attachments/" + UUID.randomUUID().toString();

            // Determine resource type based on file type
            String contentType = file.getContentType();
            String resourceType = "auto"; // auto-detect
            String folder = "auto_chat";

            if (contentType != null) {
                if (contentType.startsWith("image/")) {
                    resourceType = "image";
                } else if (contentType.startsWith("video/")) {
                    resourceType = "video";
                } else {
                    resourceType = "raw"; // for documents, audio, etc.
                }
            }

            // Upload to Cloudinary
            @SuppressWarnings("unchecked")
            Map<String, Object> uploadResult = cloudinary.uploader().upload(file.getBytes(), 
                ObjectUtils.asMap(
                    "public_id", publicId,
                    "folder", folder,
                    "resource_type", resourceType
                ));

            log.info("File uploaded to Cloudinary: {}", uploadResult.get("secure_url"));

            return uploadResult;

        } catch (IOException e) {
            log.error("Failed to upload file to Cloudinary", e);
            throw new IOException("Failed to upload file: " + e.getMessage());
        }
    }

    /**
     * Upload ảnh lên Cloudinary
     * @param file File ảnh cần upload
     * @return URL của ảnh trên Cloudinary
     */
    public String uploadImage(MultipartFile file) throws IOException {
        try {
            // Generate unique filename
            String publicId = "prescriptions/" + UUID.randomUUID().toString();

            // Upload to Cloudinary
            @SuppressWarnings("unchecked")
            Map<String, Object> uploadResult = cloudinary.uploader().upload(file.getBytes(), 
                ObjectUtils.asMap(
                    "public_id", publicId,
                    "folder", "auto_prescriptions",
                    "resource_type", "image"
                ));

            String imageUrl = (String) uploadResult.get("secure_url");
            log.info("Image uploaded to Cloudinary: {}", imageUrl);

            return imageUrl;

        } catch (IOException e) {
            log.error("Failed to upload image to Cloudinary", e);
            throw new IOException("Failed to upload image: " + e.getMessage());
        }
    }

    /**
     * Xóa ảnh từ Cloudinary (dùng khi user xóa đơn thuốc)
     * @param imageUrl URL của ảnh cần xóa
     */
    public void deleteImage(String imageUrl) {
        try {
            // Extract public_id from URL
            // https://res.cloudinary.com/xxx/image/upload/v123/auto_prescriptions/abc.jpg
            // → public_id = auto_prescriptions/abc
            String publicId = extractPublicIdFromUrl(imageUrl);
            
            if (publicId != null) {
                cloudinary.uploader().destroy(publicId, ObjectUtils.emptyMap());
                log.info("Image deleted from Cloudinary: {}", publicId);
            }
        } catch (Exception e) {
            log.error("Failed to delete image from Cloudinary: {}", imageUrl, e);
            // Don't throw exception, just log error
        }
    }

    /**
     * Upload video lên Cloudinary
     * @param file File video cần upload
     * @return URL của video trên Cloudinary
     */
    public String uploadVideo(MultipartFile file) throws IOException {
        try {
            // Generate unique filename
            String publicId = "user_guides/" + UUID.randomUUID().toString();

            // Upload to Cloudinary
            @SuppressWarnings("unchecked")
            Map<String, Object> uploadResult = cloudinary.uploader().upload(file.getBytes(), 
                ObjectUtils.asMap(
                    "public_id", publicId,
                    "folder", "auto_user_guides",
                    "resource_type", "video"
                ));

            String videoUrl = (String) uploadResult.get("secure_url");
            log.info("Video uploaded to Cloudinary: {}", videoUrl);

            return videoUrl;

        } catch (IOException e) {
            log.error("Failed to upload video to Cloudinary", e);
            throw new IOException("Failed to upload video: " + e.getMessage());
        }
    }

    /**
     * Xóa video từ Cloudinary
     * @param videoUrl URL của video cần xóa
     */
    public void deleteVideo(String videoUrl) {
        try {
            // Extract public_id from URL
            String publicId = extractPublicIdFromUrl(videoUrl);
            
            if (publicId != null) {
                cloudinary.uploader().destroy(publicId, ObjectUtils.asMap("resource_type", "video"));
                log.info("Video deleted from Cloudinary: {}", publicId);
            }
        } catch (Exception e) {
            log.error("Failed to delete video from Cloudinary: {}", videoUrl, e);
            // Don't throw exception, just log error
        }
    }

    private String extractPublicIdFromUrl(String imageUrl) {
        try {
            // Extract public_id from Cloudinary URL
            if (imageUrl == null || !imageUrl.contains("cloudinary.com")) {
                return null;
            }

            // URL format: https://res.cloudinary.com/{cloud_name}/image/upload/v{version}/{public_id}.{format}
            // or: https://res.cloudinary.com/{cloud_name}/video/upload/v{version}/{public_id}.{format}
            String[] parts = imageUrl.split("/upload/");
            if (parts.length < 2) {
                return null;
            }

            String pathAfterUpload = parts[1];
            // Remove version (v123456789/)
            pathAfterUpload = pathAfterUpload.replaceFirst("v\\d+/", "");
            // Remove extension
            int lastDot = pathAfterUpload.lastIndexOf('.');
            if (lastDot > 0) {
                pathAfterUpload = pathAfterUpload.substring(0, lastDot);
            }

            return pathAfterUpload;
        } catch (Exception e) {
            log.error("Failed to extract public_id from URL: {}", imageUrl, e);
            return null;
        }
    }
}
