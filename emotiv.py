import os
import platform
import sys
import select
from array import *
from ctypes import *
from util import *
import logging
import threading
import timeit

userID = c_uint(0)
user   = pointer(userID)
ready  = 0
state  = c_int(0)

alphaValue     = c_double(0)
lowBetaValue   = c_double(0)
highBetaValue  = c_double(0)
gammaValue     = c_double(0)
thetaValue     = c_double(0)

thetaPtr     = pointer(thetaValue)
alphaPtr     = pointer(alphaValue)
lowBetaPtr   = pointer(lowBetaValue)
highBetaPtr  = pointer(highBetaValue)
gammaPtr     = pointer(gammaValue)

channels = array('i',[DataChannelEnum.IED_P7, DataChannelEnum.IED_O1, DataChannelEnum.IED_O2, DataChannelEnum.IED_P8])

class EEG_device(threading.Thread):
    def __init__(self):
        self.streaming = False
        self.edk = self.get_edk()
        self.set_IEE()
        self.get_stimulus()
        
        if self.IEE_EngineConnect(create_string_buffer(b"Emotiv Systems-5")) != ErrorCodeEnum.EDK_OK:
            print("Emotiv Engine start up failed.")
            exit()
            
    def get_stimulus(self):
        self.imageFileList = []
        for imageFile in os.listdir(os.path.join("static", "images")):
            self.imageFileList.append(imageFile)

    def get_edk(self):
        if sys.platform.startswith('win32'):
            global msvcrt
            import msvcrt
        try:
            if sys.platform.startswith('win32'):
                libEDK = cdll.LoadLibrary(os.path.join("sdk", "edk.dll"))
                return libEDK
            elif sys.platform.startswith('linux'):
                srcDir = os.getcwd()
                if platform.machine().startswith('arm'):
                    libPath = srcDir + "/../../bin/armhf/libedk.so"
                else:
                    libPath = srcDir + "/../../bin/linux64/libedk.so"
                libEDK = CDLL(libPath)
                return libEDK
            else:
                raise Exception('System not supported.')
        except Exception as e:
            print('Error: cannot load EDK lib:', e)
            exit()

    def set_IEE(self):
        self.IEE_EngineConnect = self.edk.IEE_EngineConnect
        self.IEE_EngineConnect.restype = c_int
        self.IEE_EngineConnect.argtypes = [c_void_p]

        self.IEE_EngineGetNextEvent = self.edk.IEE_EngineGetNextEvent
        self.IEE_EngineGetNextEvent.restype = c_int
        self.IEE_EngineGetNextEvent.argtypes = [c_void_p]

        self.IEE_EmoEngineEventGetUserId = self.edk.IEE_EmoEngineEventGetUserId
        self.IEE_EmoEngineEventGetUserId.restype = c_int
        self.IEE_EmoEngineEventGetUserId.argtypes = [c_void_p , c_void_p]

        self.IEE_EmoEngineEventGetType = self.edk.IEE_EmoEngineEventGetType
        self.IEE_EmoEngineEventGetType.restype = c_int
        self.IEE_EmoEngineEventGetType.argtypes = [c_void_p]

        IEE_EmoEngineEventCreate = self.edk.IEE_EmoEngineEventCreate
        IEE_EmoEngineEventCreate.restype = c_void_p

        self.IEE_FFTSetWindowingType = self.edk.IEE_FFTSetWindowingType
        self.IEE_FFTSetWindowingType.restype = c_int
        self.IEE_FFTSetWindowingType.argtypes = [c_uint, c_void_p]

        self.IEE_GetAverageBandPowers = self.edk.IEE_GetAverageBandPowers
        self.IEE_GetAverageBandPowers.restype = c_int
        self.IEE_GetAverageBandPowers.argtypes = [c_uint, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]

        IEE_EngineDisconnect = self.edk.IEE_EngineDisconnect
        IEE_EngineDisconnect.restype = c_int
        IEE_EngineDisconnect.argtype = c_void_p

        IEE_EmoEngineEventFree = self.edk.IEE_EmoEngineEventFree
        IEE_EmoEngineEventFree.restype = c_int
        IEE_EmoEngineEventFree.argtypes = [c_void_p]

        self.eEvent = IEE_EmoEngineEventCreate()

    def start_streaming(self, callback, lapse=6):
        ready = 0
        if not self.streaming:
            self.streaming = True

        if not isinstance(callback, list):
            callback = [callback]

        start_time = timeit.default_timer()

        while self.streaming:
            state = self.IEE_EngineGetNextEvent(self.eEvent)
            if state == ErrorCodeEnum.EDK_OK:
                eventType = self.IEE_EmoEngineEventGetType(self.eEvent)
                self.IEE_EmoEngineEventGetUserId(self.eEvent, user)
                ready = 1

                if eventType == EngineEventEnum.IEE_UserAdded:
                    self.IEE_FFTSetWindowingType(userID, WindowTypeEnum.IEE_HAMMING)
                    print("User added")
                if ready == 1:
                    for i in channels:
                        if self.IEE_GetAverageBandPowers(userID, i, thetaPtr, alphaPtr, lowBetaPtr, highBetaPtr, gammaPtr) == ErrorCodeEnum.EDK_OK:
                            print("Channel: {:<7} >> Theta: {:<7.3f}, Alpha: {:<7.3f}, Low Beta: {:<7.3f}, High Beta: {:<7.3f}, Gamma: {:<7.3f}".format(parseEdkEnum(DataChannelEnum, i), thetaValue.value, alphaValue.value, lowBetaValue.value, highBetaValue.value, gammaValue.value))
                            for call in callback:
                                call(gammaValue.value)
            elif state != ErrorCodeEnum.EDK_NO_EVENT:
                print("Internal error in Emotiv Engine ! ")
            if self.kbhit():
                break
            if(lapse > 0 and timeit.default_timer() - start_time > lapse):
                self.stop()

    def kbhit(self):
        if sys.platform.startswith('win32'):
            return msvcrt.kbhit()
        else:
            dr, dw, de = select([sys.stdin], [], [], 0)
            return dr != []

    def stop(self):
        print("Stopping streaming...\nWait for buffer to flush...")
        self.streaming = False
        logging.warning('sent <s>: stopped streaming')
        '''
        IEE_EmoEngineEventFree(eEvent)
        IEE_EngineDisconnect()
        '''