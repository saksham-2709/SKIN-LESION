import { useRef } from "react";
import { Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface ImageUploadProps {
  onImageUpload: (file: File, preview: string) => void;
  currentImage?: string;
}

const ImageUpload = ({ onImageUpload, currentImage }: ImageUploadProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPEG, PNG, WEBP, GIF, or BMP)",
        variant: "destructive",
      });
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Image must be less than 10MB",
        variant: "destructive",
      });
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      onImageUpload(file, reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleClear = () => {
    onImageUpload(null as any, "");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="space-y-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {currentImage ? (
        <div className="relative rounded-lg overflow-hidden border-2 border-border">
          <img
            src={currentImage}
            alt="Preview"
            className="w-full h-auto max-h-96 object-contain bg-muted"
          />
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-2 right-2"
            onClick={handleClear}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ) : (
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-12 text-center cursor-pointer hover:border-primary transition-colors bg-card"
        >
          <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-sm font-medium mb-2">Click to upload image</p>
          <p className="text-xs text-muted-foreground">
            Supports: JPEG, PNG, WEBP, GIF, BMP (Max 10MB)
          </p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
