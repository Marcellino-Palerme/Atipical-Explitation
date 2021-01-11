//  Parts, for white balance 
//  To use, install both macros at the same time by choosing Macros --> Install Macros.  

//  Once that's done, use Macros --> White Balance and choose the folder where the 
// images live to run white balance on all images in that 
//  folder (balanced images will be saved to the new folder with name  + balanced). 

   
// White balance by Vytas Bindokas, 
macro "White balance" {
Vl_Str_Image_Format = "JPG";			// spécifications d'entrée et de sortie des images
Vl_Str_Image_Format_Sortie = "jpg";
Dialog.create("Parametres de la macro Balance des blancs ");
Dialog.addString("Format des images en entrée de macro", Vl_Str_Image_Format);
Dialog.addString("Format des images en sortie de macro", Vl_Str_Image_Format_Sortie);
Dialog.show();

Vl_Str_Image_Format = Dialog.getString();
dir = getDirectory("Choose a Directory ");
    list = getFileList(dir);
print(list.length, ' files in this folder');
 setFont("SansSerif", 24);
 start = getTime();

//setBatchMode(true); // runs up to 6 times faster

for (f=0; f<list.length; f++) {	//main files loop
        path = dir+list[f];
       // showProgress(f, list.length);
 if (!endsWith(path,"/") && endsWith(path,Vl_Str_Image_Format)) open(path);   
 print(path);
  if (nImages>=1) {

run("Colors...", "foreground=white background=black selection=yellow");
setTool("rectangle");
makeRectangle(5800, 1200, 136, 160);
//waitForUser("draw rectangle for white balance, then hit OK");

// beginning of inserted white balance macro code

ti=getTitle;
run("Set Measurements...", "  mean redirect=None decimal=3");
roiManager("add");
                if (roiManager("count")==0)
               exit("you must draw region first");
roiManager("deselect");
run("RGB Stack");
roiManager("select",0);
setSlice(1);
run("Measure");
R=getResult("Mean");   
setSlice(2);
run("Measure");
G=getResult("Mean");
setSlice(3);
run("Measure");
B=getResult("Mean");
print(B);
roiManager("reset");
run("Select None");
run("16-bit");
run("32-bit");
t=((R+G+B)/3);

setSlice(1);
dR=R-t;
if (dR<0){
run("Add...", "slice value="+abs(dR));
} else if (dR>0) {
run("Subtract...", "slice value="+abs(dR));
}

setSlice(2);
dG=G-t;
if (dG<0){
run("Add...", "slice value="+abs(dG));
} else if (dG>0) {
run("Subtract...", "slice value="+abs(dG));
}
setSlice(3);
dB=B-t;
if (dB<0){
run("Add...", "slice value="+abs(dB));
} else if (dB>0) {
run("Subtract...", "slice value="+abs(dB));
}
run("16-bit");
run("Convert Stack to RGB");
//run("Flip Vertically");
s=lastIndexOf(path, '.');
r=substring(path, 0,s);
//n=r+" balanced.jpg";
n=r+"_balanced."+ Vl_Str_Image_Format_Sortie;
save(n);
close();
selectWindow("ROI Manager");
run("Close");
selectWindow("Results");
run("Close");
selectWindow("Log");
run("Close");
//selectImage(1);
//run("Convert Stack to RGB");
//rename("original");
//selectWindow(ti);
close();

//  end white balance macro code

          } 
      }
  }

