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
        
        // Vision::Video
        void playVideo();

        // Vision::AI
        void ai();
};