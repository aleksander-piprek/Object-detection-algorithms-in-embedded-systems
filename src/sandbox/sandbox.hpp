class Sandbox
{
    public:
        Sandbox() = default;
        void play();
        
        // Vision::Image
        void imageResize();
        void imageGrayscale();
        void imageBlur();
        void imageEdgeDetection();
        void rotateImage();
        void imageProcessDetections();
        
        // Vision::Video
        void playVideo();
        void videoProcessDetections();
};