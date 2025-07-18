class Sandbox
{
    public:
        Sandbox() = default;
        void play();
    
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
};