from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from TTS.api import TTS
import streamlit as st
import pandas as pd
x = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
def clone(text ,speaker_wav , language):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    
    voice_path = "clone_result"
    if not os.path.exists(voice_path):
        os.makedirs(voice_path)
    output_wav = 'clone_result/my_voice.wav'
    print(language)
    tts.tts_to_file(text=text,
                file_path=output_wav,
                speaker_wav=speaker_wav,
                language=language)
    return output_wav


def avatar(args):
  
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                             source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)


   
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
    
    shutil.move(result, save_dir+'.mkv')
    print('The generated video is named:', save_dir+'.mkv')

    if not args.verbose:
        shutil.rmtree(save_dir)

    return save_dir+'.mkv'

    
if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)



    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    def save_uploaded_file(uploaded_file):
    # Save the uploaded file to a temporary location
        temp_location = f"temp/{uploaded_file.name}"
        with open(temp_location, "wb") as f:
            f.write(uploaded_file.read())
        return temp_location
    st.title("Clone your voice ")

    clone_form =  st.form("clone form")
    user_text = clone_form.text_input("Enter your text (Important)")
    speaker_wave = clone_form.file_uploader("Choose Wave File (Important)")
    languages_dict = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Polish': 'pl',
    'Turkish': 'tr',
    'Russian': 'ru',
    'Dutch': 'nl',
    'Czech': 'cs',
    'Arabic': 'ar',
    'Chinese': 'zh-cn',
    'Japanese': 'ja',
    'Hungarian': 'hu',
    'Korean': 'ko',
    'Hindi': 'hi'
}

    languages_list = list(languages_dict.keys())
    
    selected_lan = clone_form.selectbox("choose a language:", languages_list,index=0)

    clone_submit = clone_form.form_submit_button("start cloning")
    if clone_submit:
        if not user_text:
            clone_form.error("Please enter your text")
            st.stop()
        if not speaker_wave:
            clone_form.error("Please enter your voice")
            st.stop()
        
        language = languages_dict[selected_lan]
        folder_path = "temp"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        temp_location = save_uploaded_file(speaker_wave)
        user_wave = os.path.abspath(temp_location)
        the_cloned_voice = clone(user_text ,user_wave , language)
        st.audio(the_cloned_voice, format="audio/wav")
        shutil.rmtree(folder_path)

        

    




    st.title("Creat your avatar")

    # Create a form context
    vid_form =  st.form("video form")
    # Step 1: Choose Image or Video
    choose_image_video = vid_form.file_uploader("Choose Image or Video (Important)")

    # Step 2: Choose Eye Blink Video
    choose_eye_blink_video = vid_form.file_uploader("Choose Eye Blink Video")

    # Step 3: Checkbox for taking the same video as head pose video
    take_same_video = vid_form.checkbox("Take the same video as head pose video")

    # Step 4: Choose Head Pose Video
    choose_head_pose_video = vid_form.file_uploader("Choose Head Pose Video")

    # Step 5: Choose Wave File (Important)
    choose_wave_file = vid_form.file_uploader("Choose Wave File (Important)")

    # Button to trigger the processing function
    submit_button = vid_form.form_submit_button("creat the avatar")
    if not submit_button:
        st.stop()
    # Validate form submission
    if submit_button and choose_image_video and choose_wave_file:
        if take_same_video and not choose_eye_blink_video :
            vid_form.error("you did not choose the eye video")
            st.stop()
        folder_path = "temp"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        temp_location = save_uploaded_file(choose_image_video)
        avatar_path = os.path.abspath(temp_location)
        temp_location = save_uploaded_file(choose_wave_file)
        wave_path = os.path.abspath(temp_location)
        if choose_eye_blink_video :
            temp_location = save_uploaded_file(choose_eye_blink_video)
            eye_path = os.path.abspath(temp_location)
            args.ref_eyeblink = eye_path
        if choose_head_pose_video :
            temp_location = save_uploaded_file(choose_head_pose_video)
            head_path = os.path.abspath(temp_location)
            args.ref_pose = head_path
        elif take_same_video:
            args.ref_pose  = args.ref_eyeblink
        args.source_image = avatar_path
        args.driven_audio = wave_path
        vid=avatar(args)
        st.video(vid)
        shutil.rmtree(folder_path)

    else:
        if not choose_image_video:
            vid_form.error("Please choose Image or Video")
            st.stop()
        else : 
            vid_form.error("Please choose Wave File")
            st.stop()



