/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2012 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <jni.h>
#include "Logger.hpp"
#include "JNIUtils.hpp"
#include "PointerUtils.hpp"

jmethodID Object_getClass; // ()Ljava/lang/Class;

jmethodID Class_getComponentType; // ()Ljava/lang/Class;
jmethodID Class_newInstance; // ()Ljava/lang/Object;

jmethodID Buffer_isDirect; // ()Z
jmethodID Buffer_hasArray; // ()Z
jmethodID Buffer_array; // ()Ljava/lang/Object;

jfieldID NativePointerObject_nativePointer; // long

jclass Pointer_class; // Global reference to jcuda/Pointer class
jfieldID Pointer_buffer; // Ljava.nio.Buffer;
jfieldID Pointer_pointers; // [jcuda.NativePointerObject;
jfieldID Pointer_byteOffset; // long




/**
 * Initialize the field- and method IDs for the PointerUtils
 */
int initPointerUtils(JNIEnv *env)
{
    jclass cls = NULL;

    // Obtain the methodID for Object#getClass
    if (!init(env, cls, "java/lang/Object")) return JNI_ERR;
    if (!init(env, cls, Object_getClass, "getClass", "()Ljava/lang/Class;")) return JNI_ERR;

    // Obtain the methodID for Class#getComponentType
    if (!init(env, cls, "java/lang/Class")) return JNI_ERR;
    if (!init(env, cls, Class_getComponentType, "getComponentType", "()Ljava/lang/Class;")) return JNI_ERR;
    if (!init(env, cls, Class_newInstance,      "newInstance",      "()Ljava/lang/Object;")) return JNI_ERR;

    // Obtain the methodIDs for Buffer: isDirect, hasArray and array
    if (!init(env, cls, "java/nio/Buffer")) return JNI_ERR;
    if (!init(env, cls, Buffer_isDirect, "isDirect", "()Z"                 )) return JNI_ERR;
    if (!init(env, cls, Buffer_hasArray, "hasArray", "()Z"                 )) return JNI_ERR;
    if (!init(env, cls, Buffer_array,    "array",    "()Ljava/lang/Object;")) return JNI_ERR;


    // Obtain the fieldIDs of the NativePointerObject class
    if (!init(env, cls, "jcuda/NativePointerObject")) return JNI_ERR;
    if (!init(env, cls, NativePointerObject_nativePointer, "nativePointer", "J")) return JNI_ERR;

    // Obtain the fieldIDs of the Pointer class
    if (!init(env, cls, "jcuda/Pointer")) return JNI_ERR;
    Pointer_class = (jclass)env->NewGlobalRef(cls);
    if (Pointer_class == NULL)
    {
        return JNI_ERR;
    }
    if (!init(env, cls, Pointer_buffer,        "buffer",        "Ljava/nio/Buffer;"            )) return JNI_ERR;
    if (!init(env, cls, Pointer_pointers,      "pointers",      "[Ljcuda/NativePointerObject;" )) return JNI_ERR;
    if (!init(env, cls, Pointer_byteOffset,    "byteOffset",    "J"                            )) return JNI_ERR;

    return JNI_VERSION_1_4;
}


/**
 * If the given PointerData is NULL, an OutOfMemoryError is thrown
 * and NULL is returned.
 * Otherwise, this function tries to initialize the PointerData with
 * the given Java NativePointerObject. If the initialization fails,
 * the PointerData is deleted and NULL is returned.
 * Otherwise, the initialized pointer data is returned.
 */
PointerData *validatePointerData(JNIEnv *env, jobject nativePointerObject, PointerData *pointerData)
{
    if (pointerData == NULL)
    {
        ThrowByName(env, "java/lang/OutOfMemoryError",
            "Out of memory while creating pointer data");
        return NULL;
    }
    if (!pointerData->init(env, nativePointerObject))
    {
        delete pointerData;
        return NULL;
    }
    return pointerData;
}

/**
 * Initializes a PointerData with the data from the given
 * Java NativePointerObject.
 *
 * This will return an implementation of the PointerData
 * class, depending on the type of the given Java object:
 *
 * - If the given object is 'NULL' or is NO Java Pointer
 *   (but only a NativePointerObject)
 *   then a NativePointerObjectPointerData is returned
 *
 * Otherwise the given object is a valid Java Pointer object.
 *
 * - If the 'pointers' array of the given Pointer object is non-NULL,
 *   then a PointersArrayPointerData is returned
 *
 * - If the 'buffer' of the given Pointer is non-NULL, then
 *   a PointerData object will be returned depending on
 *   the properties of the buffer:
 *   - If the buffer is direct, then a DirectBufferPointerData is returned
 *   - If the buffer has an array, then a ArrayBufferPointerData is returned
 *
 * - Otherwise, a NativePointerData will be returned, which has its
 *   nativePointer value and byteOffset set according to the values
 *   in the given Pointer
 *
 * In any case, if an Exception occurs or the initialization of
 * the PointerData fails, then NULL is returned.
 */
PointerData* initPointerData(JNIEnv *env, jobject nativePointerObject)
{
    Logger::log(LOG_DEBUGTRACE, "Initializing pointer data for Java NativePointerObject %p\n", nativePointerObject);

    // If the given object is 'NULL', then return a NativePointerObjectPointerData
    if (nativePointerObject == NULL)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing NativePointerObjectPointerData\n");
        NativePointerObjectPointerData *pointerData = new NativePointerObjectPointerData();
        return validatePointerData(env, nativePointerObject, pointerData);
    }

    // If the object is no Pointer (but only a NativePointerObject), 
	// then return a NativePointerObjectPointerData
    jboolean isPointer = env->IsInstanceOf(nativePointerObject, Pointer_class);
    if (!isPointer)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing NativePointerObjectPointerData\n");
        NativePointerObjectPointerData *pointerData = new NativePointerObjectPointerData();
        return validatePointerData(env, nativePointerObject, pointerData);
    }

    // If the Pointer contains a 'pointers' array, then return
    // a PointersArrayPointerData
    jobjectArray pointersArray = (jobjectArray)env->GetObjectField(nativePointerObject, Pointer_pointers);
    if (pointersArray != NULL)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing PointersArrayPointerData\n");
        PointersArrayPointerData *pointerData = new PointersArrayPointerData();
        return validatePointerData(env, nativePointerObject, pointerData);
    }

    // Check if the Pointer contains a buffer
    jobject buffer = env->GetObjectField(nativePointerObject, Pointer_buffer);
    if (buffer != NULL)
    {
        // If the buffer is direct, return a DirectBufferPointerData
        jboolean isDirect = env->CallBooleanMethod(buffer, Buffer_isDirect);
        if (env->ExceptionCheck())
        {
            return NULL;
        }
        if (isDirect==JNI_TRUE)
        {
            Logger::log(LOG_DEBUGTRACE, "Initializing DirectBufferPointerData for\n");
            DirectBufferPointerData *pointerData = new DirectBufferPointerData();
            return validatePointerData(env, nativePointerObject, pointerData);
        }

        // If the buffer has an array, return a ArrayBufferPointerData
        jboolean hasArray = env->CallBooleanMethod(buffer, Buffer_hasArray);
        if (env->ExceptionCheck())
        {
            return NULL;
        }
        if (hasArray==JNI_TRUE)
        {
            Logger::log(LOG_DEBUGTRACE, "Initializing ArrayBufferPointerData\n");
            ArrayBufferPointerData *pointerData = new ArrayBufferPointerData();
            return validatePointerData(env, nativePointerObject, pointerData);
        }

        // The buffer is neither direct nor has an array - should have
        // been checked on Java side
        Logger::log(LOG_ERROR, "Buffer is neither direct nor has an array\n");
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Buffer is neither direct nor has an array");
        return NULL;
    }

    // At this point, the given object must be a Pointer, containing
	// a (possibly NULL) nativePointer and a (possibly 0) byteOffset
    Logger::log(LOG_DEBUGTRACE, "Initializing NativePointerData\n");
    NativePointerData *pointerData = new NativePointerData();
    return validatePointerData(env, nativePointerObject, pointerData);
}


/**
 * Releases the given PointerData by calling PointerData::release, 
 * and deletes the PointerData object.
 *
 * The effect of the PointerData::release method is depending on 
 * the type of the pointerData:
 *
 * - For a NativePointerData, nothing has to be done
 * - For a PointersArrayPointerData
 * - For a DirectBufferPointerData, nothing has to be done
 * - For a ArrayBufferPointerData, the primitve array is released
 *   using the given mode (JNI_COMMIT, JNI_ABORT or 0)
 *
 * If the given PointerData is NULL, then nothing is done.
 *
 * The method returns whether the respective operation succeeded.
 */
bool releasePointerData(JNIEnv *env, PointerData* &pointerData, jint mode)
{
	if (pointerData == NULL) return true;
    if (!pointerData->release(env, mode)) return false;
    delete pointerData;
    pointerData = NULL;
    return true;
}




/**
 * Returns whether the given buffer is a direct byte buffer
 */
bool isDirectByteBuffer(JNIEnv *env, jobject buffer)
{
    if (buffer == NULL)
    {
        return false;
    }
    jboolean isDirect = env->CallBooleanMethod(buffer, Buffer_isDirect);
    if (env->ExceptionCheck())
    {
        return false;
    }
    if (isDirect == JNI_TRUE)
    {
        return true;
    }
    return false;
}


/**
 * Returns whether the given pointer object is either a pointer
 * with a nativePointer value that is not null, or a pointer to
 * a direct ByteBuffer
 */
bool isPointerBackedByNativeMemory(JNIEnv *env, jobject object)
{
    if (object == NULL)
    {
        return false;
    }
    jlong nativePointer = env->GetLongField(object, NativePointerObject_nativePointer);
    if (nativePointer != NULL)
    {
        return true;
    }
    jboolean isPointer = env->IsInstanceOf(object, Pointer_class);
    if (isPointer)
    {
        jobject buffer = env->GetObjectField(object, Pointer_buffer);
        return isDirectByteBuffer(env, buffer);
    }
    return false;
}




/**
 * Set the nativePointer in the given Java NativePointerObject to the given
 * pointer.
 */
void setNativePointerValue(JNIEnv *env, jobject nativePointerObject, jlong pointer)
{
    if (nativePointerObject == NULL)
    {
        return;
    }
    env->SetLongField(nativePointerObject, NativePointerObject_nativePointer, pointer);
}


/**
 * Returns the nativePointer of the given
 * Java NativePointerObject
 */
void* getNativePointerValue(JNIEnv *env, jobject nativePointerObject)
{
    if (nativePointerObject == NULL)
    {
        return NULL;
    }
    jlong pointer = env->GetLongField(nativePointerObject, NativePointerObject_nativePointer);
    return (void*)pointer;
}


/**
 * Set the nativePointer in the given Java Pointer object to the given
 * pointer. The byteOffset will be set to 0.
 */
void setPointer(JNIEnv *env, jobject pointerObject, jlong pointer)
{
    if (pointerObject == NULL)
    {
        return;
    }
    env->SetLongField(pointerObject, NativePointerObject_nativePointer, pointer);
    env->SetLongField(pointerObject, Pointer_byteOffset, 0);
}


/**
 * Returns the native pointer that is described by the given
 * Java Pointer object, i.e. its nativePointer plus its
 * byteOffset.
 */
void* getPointer(JNIEnv *env, jobject pointerObject)
{
    if (pointerObject == NULL)
    {
        return NULL;
    }
    jlong startPointer = env->GetLongField(pointerObject, NativePointerObject_nativePointer);
    jlong byteOffset = env->GetLongField(pointerObject, Pointer_byteOffset);
    jlong pointer = startPointer+byteOffset;
    return (void*)pointer;
}

