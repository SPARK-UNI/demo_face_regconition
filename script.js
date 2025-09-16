class FaceRecognition {
    constructor() {
        this.video = document.getElementById('video');
        this.startBtn = document.getElementById('startBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.result = document.getElementById('result');
        this.status = document.getElementById('status');
        this.statusDot = document.querySelector('.status-dot');
        this.loading = document.getElementById('loading');
        this.placeholder = document.getElementById('placeholder');
        
        this.stream = null;
        this.isProcessing = false;
        
        this.init();
    }
    
    init() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.predictBtn.addEventListener('click', () => this.recognizeFace());
        
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.updateResult('Browser không hỗ trợ camera', 'error');
            this.startBtn.disabled = true;
        }
    }
    
    async startCamera() {
        try {
            this.startBtn.disabled = true;
            this.updateResult('Đang khởi động camera...', '');
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.placeholder.classList.add('hidden');
            
            // Wait for video to be ready
            this.video.onloadedmetadata = () => {
                this.updateStatus(true);
                this.predictBtn.disabled = false;
                this.stopBtn.disabled = false;
                this.updateResult('Camera đã sẵn sàng', 'success');
            };
            
        } catch (error) {
            console.error('Camera error:', error);
            this.handleCameraError(error);
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.srcObject = null;
        this.placeholder.classList.remove('hidden');
        
        this.startBtn.disabled = false;
        this.predictBtn.disabled = true;
        this.stopBtn.disabled = true;
        
        this.updateStatus(false);
        this.updateResult('Camera đã tắt', '');
    }
    
    async recognizeFace() {
        if (this.isProcessing || !this.stream) return;
        
        this.isProcessing = true;
        this.predictBtn.disabled = true;
        this.showLoading(true);
        
        try {
            const imageData = this.captureFrame();
            const prediction = await this.sendToAPI(imageData);
            
            this.updateResult(prediction, 'success');
            
        } catch (error) {
            console.error('Recognition error:', error);
            this.updateResult('Lỗi khi nhận dạng khuôn mặt', 'error');
        } finally {
            this.isProcessing = false;
            this.predictBtn.disabled = false;
            this.showLoading(false);
        }
    }
    
    captureFrame() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        
        context.drawImage(this.video, 0, 0);
        
        return canvas.toDataURL('image/jpeg', 0.8);
    }
    
    async sendToAPI(imageData) {
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const data = await response.json();
            return data.prediction;
            
        } catch (error) {
            throw new Error('Không thể kết nối đến server');
        }
    }
    updateStatus(online) {
        if (online) {
            this.status.innerHTML = '<span class="status-dot online"></span>Camera Online';
            this.statusDot.classList.remove('offline');
            this.statusDot.classList.add('online');
        } else {
            this.status.innerHTML = '<span class="status-dot offline"></span>Camera Offline';
            this.statusDot.classList.remove('online');
            this.statusDot.classList.add('offline');
        }
    }
    
    updateResult(text, type) {
        this.result.textContent = text;
        this.result.className = `result-text ${type}`;
    }
    
    showLoading(show) {
        this.loading.classList.toggle('hidden', !show);
        this.result.style.display = show ? 'none' : 'block';
    }
    
    handleCameraError(error) {
        let errorMessage = 'Không thể truy cập camera';
        
        if (error.name === 'NotAllowedError') {
            errorMessage = 'Vui lòng cho phép truy cập camera';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'Không tìm thấy camera';
        } else if (error.name === 'NotSupportedError') {
            errorMessage = 'Trình duyệt không hỗ trợ camera';
        }
        
        this.updateResult(errorMessage, 'error');
        this.startBtn.disabled = false;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.faceRecognition = new FaceRecognition();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.faceRecognition && window.faceRecognition.stream) {
        window.faceRecognition.stopCamera();
    }
});