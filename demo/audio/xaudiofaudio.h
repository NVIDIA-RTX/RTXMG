/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cstring>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <FAudio.h>
#include <FAudioFX.h>

/////////////////
// imitate the smallest subset of the XAudio2 COM interface this sample uses, on top of FAudio

typedef FAudioBuffer XAUDIO2_BUFFER;
typedef FAudioBufferWMA XAUDIO2_BUFFER_WMA;
typedef FAudioVoiceSends XAUDIO2_VOICE_SENDS;
typedef FAudioVoiceState XAUDIO2_VOICE_STATE;
typedef FAudioSendDescriptor XAUDIO2_SEND_DESCRIPTOR;
typedef FAudioEffectChain XAUDIO2_EFFECT_CHAIN;
typedef FAudioProcessor XAUDIO2_PROCESSOR;
typedef FAudioWaveFormatEx WAVEFORMATEX;
typedef FAudioVoiceCallback IXAudio2VoiceCallback;
#define XAUDIO2_COMMIT_NOW FAUDIO_COMMIT_NOW
#define XAUDIO2_SEND_USEFILTER FAUDIO_SEND_USEFILTER
#define XAUDIO2_DEFAULT_FREQ_RATIO FAUDIO_DEFAULT_FREQ_RATIO
#define XAUDIO2_LOOP_INFINITE FAUDIO_LOOP_INFINITE
#define XAUDIO2_NO_LOOP_REGION FAUDIO_NO_LOOP_REGION
#define XAUDIO2_END_OF_STREAM FAUDIO_END_OF_STREAM
#define XAUDIO2_DEFAULT_CHANNELS FAUDIO_DEFAULT_CHANNELS
#define XAUDIO2_DEFAULT_SAMPLERATE FAUDIO_DEFAULT_SAMPLERATE
#define XAUDIO2_DEFAULT_PROCESSOR FAUDIO_DEFAULT_PROCESSOR
class IXAudio2;
class IXAudio2SourceVoice;
class IXAudio2MasteringVoice;

#define GRAB_VOICE_OBJ(P) ((P)->_GetFAudioVoidPtr()) /* fudge because of how I've implemented the shim */

/////////////////
// ... plus some basic win32-alikes to make the shim easier to use transparently
#ifndef _WIN32
typedef uint32_t HRESULT;
typedef uint8_t BYTE;
#define S_OK 0
// FAudio returns 0 on success and an unsigned error code otherwise.
#define FAILED(R) ((HRESULT)(R) != 0)

#define CoInitializeEx(FOO,BAR) S_OK /* not really using COM, pretend we're fine */
#define COINIT_MULTITHREADED

#define WAVE_FORMAT_PCM FAUDIO_FORMAT_PCM
#endif // _WIN32

/////////////////
// here's the core of the fakey XAudio2 COM interface
class IXAudio2MasteringVoice
{
public:
  static IXAudio2MasteringVoice* _XAudio2MasteringVoiceCreateFromFAudioMasteringVoice(FAudioMasteringVoice *faudiomv)
  {
    IXAudio2MasteringVoice* rtn = new IXAudio2MasteringVoice;
    rtn->_fa_voice = faudiomv;
    return rtn;
  };
  FAudioVoice* _GetFAudioVoidPtr() {return _fa_voice;};

  HRESULT SetVolume(float Volume, uint32_t OperationSet=XAUDIO2_COMMIT_NOW) { return FAudioVoice_SetVolume(_fa_voice,Volume,OperationSet); };
  void GetVolume(float *pVolume) { FAudioVoice_GetVolume(_fa_voice,pVolume); };

  void DestroyVoice() noexcept { FAudioVoice_DestroyVoice(_fa_voice); _fa_voice = NULL; }

private:
  FAudioMasteringVoice* _fa_voice;
};

class IXAudio2SourceVoice
{
public:
  static IXAudio2SourceVoice* _XAudio2SourceVoiceCreateFromFAudioSourceVoice(FAudioSourceVoice *faudiosv)
  {
    IXAudio2SourceVoice* rtn = new IXAudio2SourceVoice;
    rtn->_fa_voice = faudiosv;
    return rtn;
  };

  void DestroyVoice() { FAudioVoice_DestroyVoice(_fa_voice); _fa_voice = NULL; };
  HRESULT Start(uint32_t Flags=0, uint32_t OperationSet=XAUDIO2_COMMIT_NOW) { return FAudioSourceVoice_Start(_fa_voice,Flags,OperationSet); };
  HRESULT Stop(uint32_t Flags=0, uint32_t OperationSet=XAUDIO2_COMMIT_NOW) { return FAudioSourceVoice_Stop(_fa_voice,Flags,OperationSet); };
  HRESULT ExitLoop(uint32_t OperationSet=XAUDIO2_COMMIT_NOW) { return FAudioSourceVoice_ExitLoop(_fa_voice,OperationSet); };
  HRESULT SetVolume(float Volume, uint32_t OperationSet=XAUDIO2_COMMIT_NOW) { return FAudioVoice_SetVolume(_fa_voice,Volume,OperationSet); };
  void GetVolume(float *pVolume) { FAudioVoice_GetVolume(_fa_voice,pVolume); };
  HRESULT SetFrequencyRatio(float Ratio, uint32_t OperationSet=XAUDIO2_COMMIT_NOW) { return FAudioSourceVoice_SetFrequencyRatio(_fa_voice,Ratio,OperationSet); };
  void GetFrequencyRatio(float *pRatio) { FAudioSourceVoice_GetFrequencyRatio(_fa_voice,pRatio); };
  HRESULT SubmitSourceBuffer(const XAUDIO2_BUFFER* pBuffer, const XAUDIO2_BUFFER_WMA* pBufferWMA = NULL) { return FAudioSourceVoice_SubmitSourceBuffer(_fa_voice,pBuffer,pBufferWMA); };
  HRESULT FlushSourceBuffers() { return FAudioSourceVoice_FlushSourceBuffers(_fa_voice); };
void GetState(XAUDIO2_VOICE_STATE* pVoiceState, uint32_t Flags=0) { FAudioSourceVoice_GetState(_fa_voice,pVoiceState,Flags); };

private:
  FAudioSourceVoice* _fa_voice;
};

class IXAudio2
{
public:
  static IXAudio2* _XAudio2CreateFromFAudio(FAudio *faudio)
  {
    IXAudio2* rtn = new IXAudio2;
    rtn->_faudio = faudio;
    return rtn;
  };
  unsigned long Release()
  { uint32_t refcount = FAudio_Release(_faudio); if (!refcount) _faudio = nullptr; return refcount; };

  HRESULT CreateSourceVoice(
      IXAudio2SourceVoice **ppSourceVoice,
      const WAVEFORMATEX *pSourceFormat,
      uint32_t Flags = 0,
      float MaxFrequencyRatio = XAUDIO2_DEFAULT_FREQ_RATIO,
      IXAudio2VoiceCallback *pCallback = NULL,
      const XAUDIO2_VOICE_SENDS *pSendList = NULL,
      const XAUDIO2_EFFECT_CHAIN *pEffectChain = NULL)
  {
    FAudioSourceVoice* faudiosv = nullptr;
    HRESULT rtn = FAudio_CreateSourceVoice(_faudio,&faudiosv,pSourceFormat,Flags,MaxFrequencyRatio,pCallback,pSendList,pEffectChain);
    *ppSourceVoice = rtn == 0 ? IXAudio2SourceVoice::_XAudio2SourceVoiceCreateFromFAudioSourceVoice(faudiosv) : nullptr;
    return rtn;
  }
  HRESULT CreateMasteringVoice(
      IXAudio2MasteringVoice **ppMasteringVoice,
      uint32_t InputChannels = XAUDIO2_DEFAULT_CHANNELS,
      uint32_t InputSampleRate = XAUDIO2_DEFAULT_SAMPLERATE,
      uint32_t Flags = 0,
      uint32_t DeviceIndex = 0,
      const XAUDIO2_EFFECT_CHAIN *pEffectChain = NULL)
  {
    FAudioMasteringVoice* faudiomv = nullptr;
    HRESULT rtn = FAudio_CreateMasteringVoice(_faudio,&faudiomv,InputChannels,InputSampleRate,Flags,DeviceIndex,pEffectChain);
    *ppMasteringVoice = rtn == 0 ? IXAudio2MasteringVoice::_XAudio2MasteringVoiceCreateFromFAudioMasteringVoice(faudiomv) : nullptr;
    return rtn;
  }

private:
  FAudio* _faudio;
};

static HRESULT XAudio2Create(
  IXAudio2 **ppXAudio2,
  uint32_t Flags = 0,
  XAUDIO2_PROCESSOR XAudio2Processor = XAUDIO2_DEFAULT_PROCESSOR
)
{
  FAudio* faudio = nullptr;
  HRESULT rtn = FAudioCreate(&faudio,Flags,XAudio2Processor);
  *ppXAudio2 = rtn == 0 ? IXAudio2::_XAudio2CreateFromFAudio(faudio) : nullptr;
  return rtn;
};


