class Example
{
    public:
        Example() = default;
        void run();
    
    private:
        // ImageInput
        void imageResize();
        void imageGrayscale();
        void imageBlur();
        void imageEdgeDetection();
        void rotateImage();
        void imageInference();
        
        // VideoInput
        void playVideo();
        void videoInference();

        // CameraInput
        void playCamera();
        void cameraInference();
};