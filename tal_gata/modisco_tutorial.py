###
# THIS CODE IS UGLY. IT'S JUST HERE FOR THE PURPOSE OF A DEMO.
# CLEAN CODE WILL BE PREPARED AND RELEASED WHEN MODISCO IS
# READY FOR PRIME TIME.
# contact: avanti.shrikumar@gmail.com
###

from __future__ import division;
from __future__ import print_function;
from __future__ import absolute_import;
import sys, os;
import numpy as np;
import time
from scipy import signal
import itertools
from collections import OrderedDict, namedtuple, defaultdict, Counter
import util;
from util import CROSSC_NORMFUNC
from avutils import file_processing as fp
import identifyPeaks;   
import deeplift
import theano

core_importance_track='core_importance_track'

def dna_rev_comp_func(arr):
    assert len(arr.shape)==2, arr.shape
    assert arr.shape[0]==4;
    return arr[::-1,::-1];
def reverseFunc(arr):
    return arr[:,::-1];
def identity(arr):
    return arr
class RevCompWithDNArowsSubset(object):
    __name__ = "RevCompWithDNArowsSubset" #for json saving...
    def __init__(self, dnaRowsStart, dnaRowsEnd):
        self.dnaRowsStart=dnaRowsStart;
        self.dnaRowsEnd=dnaRowsEnd;
    def __call__(self,arr):
        toReturn = np.zeros(arr.shape);
        assert self.dnaRowsEnd-self.dnaRowsStart==4;
        assert arr.shape[0] >= self.dnaRowsEnd, arr.shape;
        if (self.dnaRowsStart > 0):
            toReturn[:self.dnaRowsStart,:]=arr[:self.dnaRowsStart,::-1];
        toReturn[self.dnaRowsStart:self.dnaRowsEnd,:]=\
            dna_rev_comp_func(arr[self.dnaRowsStart:self.dnaRowsEnd])
        if (arr.shape[0] > self.dnaRowsEnd):
            toReturn[self.dnaRowsEnd:,:]=arr[self.dnaRowsEnd,::-1];
        return toReturn;

class DataTrack(object):
    __slots__ = ["data", "revCompData", "pseudocount", "effectiveStride"
                 , "effectiveWidth", "layerFromAbove", "fillValue"
                 , "minVisibility"]
    JsonKeys = util.enum(data="data"
                        , revCompData="revCompData"
                        , pseudocount="pseudocount"
                        , effectiveStride="effectiveStride"
                        , effectiveWidth="effectiveWidth"
                        , layerFromAbove="layerFromAbove"
                        , fillValue="fillValue"
                        , minVisibility="minVisibility");
    def __init__(self, data, revCompData, pseudocount, effectiveStride
                     , effectiveWidth, layerFromAbove, fillValue
                     , minVisibility):
        """
            effectiveStride and effectiveWidth will be 1 for most purposes...they are only
                relevant when your ModiscoMotif object has augmented data tracks that are of
                different lengths - eg: the DeepLIFT scores of a convolutional layer
                and the sequence underlying the convolutional layer. In that case,
                assuming that the grammar object was created on the convolutional layer
                deepLIFT scores, then the effective stride and effective width of
                the sequence track must be set to the stride and width of the conv layer.
        """
        self.data = data;
        self.revCompData = revCompData;
        self.pseudocount = pseudocount;
        self.effectiveStride = effectiveStride;
        self.effectiveWidth = effectiveWidth;
        self.layerFromAbove=layerFromAbove;
        self.fillValue=fillValue;
        self.minVisibility=minVisibility;
    def reverseComplement(self):
        return DataTrack(
                data=self.revCompData
                , revCompData=self.data
                , pseudocount=self.pseudocount
                , effectiveStride=self.effectiveStride
                , effectiveWidth=self.effectiveWidth
                , layerFromAbove=self.layerFromAbove
                , fillValue=self.fillValue
                , minVisibility=self.minVisibility)
    def getJsonableObject(self):
        return {DataTrack.JsonKeys.data:self.data.tolist()
                , DataTrack.JsonKeys.revCompData:self.revCompData.tolist()
                , DataTrack.JsonKeys.pseudocount:self.pseudocount
                , DataTrack.JsonKeys.effectiveStride:self.effectiveStride
                , DataTrack.JsonKeys.effectiveWidth:self.effectiveWidth
                , DataTrack.JsonKeys.layerFromAbove:self.layerFromAbove
                , DataTrack.JsonKeys.fillValue:self.fillValue
                , DataTrack.JsonKeys.minVisibility:self.minVisibility};
    @classmethod
    def loadFromJsonableObject(cls, jsonableObject):
        return DataTrack(data=np.array(jsonableObject[DataTrack.JsonKeys.data])
                        , revCompData=np.array(jsonableObject[DataTrack.JsonKeys.revCompData])
                        , pseudocount=jsonableObject[DataTrack.JsonKeys.pseudocount]
                        , effectiveStride=jsonableObject[DataTrack.JsonKeys.effectiveStride]
                        , effectiveWidth=jsonableObject[DataTrack.JsonKeys.effectiveWidth]
                        , layerFromAbove=jsonableObject[DataTrack.JsonKeys.layerFromAbove]
                        , fillValue=jsonableObject[DataTrack.JsonKeys.fillValue]
                        , minVisibility=jsonableObject[DataTrack.JsonKeys.minVisibility]);

class ModiscoMotif(object):
    core_importance_track = core_importance_track;
    JsonKeys = util.enum(num_underlying_observations="num_underlying_observations"
                        , totalObservationsEver="totalObservationsEver"
                        , summedDataTracks="summedDataTracks"
                        , minPseudocount="minPseudocount"
                        , pseudocountFrac="pseudocountFrac"
                        , derivedClass="derivedClass");
    def __init__(self, num_underlying_observations
                     , totalObservationsEver
                     , summedDataTracks
                     , minPseudocount=0
                     , pseudocountFrac=0.1):
        if (isinstance(num_underlying_observations, int) or isinstance(num_underlying_observations, float)):
            self.num_underlying_observations =\
                np.zeros((summedDataTracks[ModiscoMotif.core_importance_track].data.shape[1],), dtype="float")\
                         +num_underlying_observations;
        else:
            self.num_underlying_observations = util.npArrayIfList(num_underlying_observations);
        self.totalObservationsEver = totalObservationsEver;
        self.minPseudocount = minPseudocount;
        self.pseudocountFrac = pseudocountFrac; 
        self.pseudocountMultiplier = np.maximum(self.minPseudocount,
                                np.floor(self.pseudocountFrac*np.max(self.num_underlying_observations)));
        for dataTrack in summedDataTracks.values():
            assert util.assertIsType(dataTrack, DataTrack, "dataTrack");
        self.summedDataTracks = {};
        self.normalisedDataTracks = {};
        self.revCompedNormalisedDataTracks = {};
        for key, dataTrack in summedDataTracks.items():
            self.addSummedDataTrack(key, dataTrack);

    def getJsonableObject(self):
        theClass = self.__class__.__name__
        return {ModiscoMotif.JsonKeys.num_underlying_observations: self.num_underlying_observations.tolist()
                , ModiscoMotif.JsonKeys.totalObservationsEver: self.totalObservationsEver
                , ModiscoMotif.JsonKeys.summedDataTracks: OrderedDict(
                                                        [(key, self.summedDataTracks[key].getJsonableObject())
                                                          for key in self.summedDataTracks])
                , ModiscoMotif.JsonKeys.minPseudocount: self.minPseudocount
                , ModiscoMotif.JsonKeys.pseudocountFrac: self.pseudocountFrac
                , ModiscoMotif.JsonKeys.derivedClass: theClass};

    @staticmethod
    def saveListOfModiscoMotifsToJson(jsonFileName, listOfModiscoMotifs):
        import json
        jsonifiedModiscoMotifs = [x.getJsonableObject() for x in listOfModiscoMotifs];
        fp.write_to_file(jsonFileName, json.dumps(jsonifiedModiscoMotifs));

    @staticmethod
    def loadListOfModiscoMotifsFromJson(jsonFileName):
        jsonifiedModiscoMotifs = util.parseJsonFile(jsonFileName) 
        return [ModiscoMotif.loadSingleModiscoMotifOfArbitraryClass(x) for x in jsonifiedModiscoMotifs];
 
    @staticmethod
    def loadSingleModiscoMotifOfArbitraryClass(jsonableObject):
        """
            Will figure out the appropriate subclass to call for loading
        """
        theClass = eval(jsonableObject[ModiscoMotif.JsonKeys.derivedClass]); 
        return theClass.loadFromJsonableObject(jsonableObject);
        
    @classmethod
    def loadFromJsonableObject(cls, jsonableObject):
        return cls(**cls.getLoadingKwargsFromJsonableObject(jsonableObject));

    @classmethod
    def getLoadingKwargsFromJsonableObject(cls, jsonableObject):
        summedDataTracksJsonable = jsonableObject[ModiscoMotif.JsonKeys.summedDataTracks]
        summedDataTracks = OrderedDict([
                            (key,DataTrack.loadFromJsonableObject(summedDataTracksJsonable[key]))
                            for key in summedDataTracksJsonable.keys()])
        return {ModiscoMotif.JsonKeys.num_underlying_observations :\
                 jsonableObject[ModiscoMotif.JsonKeys.num_underlying_observations]
                , ModiscoMotif.JsonKeys.totalObservationsEver :\
                 jsonableObject[ModiscoMotif.JsonKeys.totalObservationsEver]
                , ModiscoMotif.JsonKeys.summedDataTracks : summedDataTracks
                , ModiscoMotif.JsonKeys.minPseudocount :\
                 jsonableObject[ModiscoMotif.JsonKeys.minPseudocount]
                , ModiscoMotif.JsonKeys.pseudocountFrac :\
                 jsonableObject[ModiscoMotif.JsonKeys.pseudocountFrac]}; 
        

    def getRevCompModiscoMotif(self):
        grammar = ModiscoMotif(
                   num_underlying_observations=
                    self.num_underlying_observations[::-1]
                   , totalObservationsEver=self.totalObservationsEver
                   , summedDataTracks={}
                   , minPseudocount=0
                   , pseudocountFrac=0.1); 
        grammar.summedDataTracks=\
         OrderedDict([(key,val.reverseComplement()) for (key,val) in
                       self.summedDataTracks.items()])
        grammar.normalisedDataTracks=self.revCompedNormalisedDataTracks;
        grammar.revCompedNormalisedDataTracks=self.normalisedDataTracks;
        return grammar;

    def getRange(self, start, end):
        grammar = ModiscoMotif(
                   num_underlying_observations=\
                    self.num_underlying_observations[start:end]
                  , totalObservationsEver=self.totalObservationsEver
                  , summedDataTracks={key:
                     DataTrack(
                      data=dataTrack.data[:,start:end]
                      , revCompData=dataTrack.data[:,
                        (dataTrack.data.shape[1]-end):(dataTrack.data.shape[1]-start)]
                      , pseudocount=dataTrack.pseudocount
                      , effectiveStride=dataTrack.effectiveStride
                      , effectiveWidth=dataTrack.effectiveWidth)
                    for (key, dataTrack) in self.summedDataTracks.items()}
                  , minPseudocount=0
                  , pseudocountFrac=0.1); 
        return grammar;

    @property
    def grammarArray(self):
        print(".grammarArray is deprecated; "
              "use normedCoreDeepLIFTtrack instead");
        return self.normedCoreDeepLIFTtrack;
    @property
    def normedCoreDeepLIFTtrack(self):
        return self.getNormalisedDataTrack(self.core_importance_track);
    @property
    def summedModiscoMotif(self):
        print(".summedModiscoMotif is deprecated; "
              "use summedCoreDeepLIFTtrack instead");
        return self.summedCoreDeepLIFTtrack;

    @property
    def summedCoreDeepLIFTtrack(self):
        return self.get_summed_data_track(self.core_importance_track);

    def getNormalisedDataTrack(self, key):
        return self.normalisedDataTracks[key];

    def get_summed_data_track(self, key):
        return self.summedDataTracks[key].data;

    def getRevCompedNormalisedDataTrack(self, key):
        return self.revCompedNormalisedDataTracks[key];

    def getRevCompedSummedDataTrack(self, key):
        return self.summedDataTracks[key].revCompData;

    @staticmethod
    def transformNumObsAccordingToWidthAndStride(numObsArr, effectiveStride
                                                 , effectiveWidth
                                                 , layerFromAbove
                                                 , minVisibility):
        #I believe layerFromAbove is a boolean indicating whether
        #this layer lay above or below the layer that was used to
        #initialize the seqlets. WffectiveStride/width will be
        #used differently, accordingly...
        #ugh this whole thing needs to be refactored to make it less kludgy
        if (layerFromAbove==False):
            #repeats entries of array, spreading
            #them out according to effectiveStride/effectiveWidth
            newObsArrLen=(len(numObsArr)-1)*effectiveStride + effectiveWidth
            newObsArr = np.ones(newObsArrLen)*-1;
            for (idx, numObs) in enumerate(numObsArr):
                startIdx = idx*effectiveStride 
                endIdx = startIdx + effectiveWidth
                maximums = np.maximum(numObs,newObsArr[startIdx:endIdx])
                newObsArr[startIdx:endIdx] = maximums; 
        else:
            assert effectiveStride==1; #only handling stride of 1 for now
            positionsWithFullVisibility=len(numObsArr)-(effectiveWidth-1)
            positionsWithPartialVisibility=2*(effectiveWidth-minVisibility)
            newObsArrLen=positionsWithFullVisibility+\
                         positionsWithPartialVisibility;
            newObsArr = np.ones(newObsArrLen)*-1;
            for (idx, numObs) in enumerate(numObsArr):
                #find the indexes in conv layer such that the neuron in
                #the conv layer sees idx and also satisfies the
                #minVisibility constraint assuming idx is the left
                #boundary of a region or the right boundary of a region
                convIdx_assumeIdxIsLeftBoundary = idx
                convIdx_assumeIdxIsRightBoundary =\
                 idx+(1+effectiveWidth-2*(minVisibility))
                #the +1 for endIdx is in order to satisfy array slicing
                startIdx, endIdx = min(convIdx_assumeIdxIsLeftBoundary,
                                       convIdx_assumeIdxIsRightBoundary),\
                                   max(convIdx_assumeIdxIsRightBoundary,
                                       convIdx_assumeIdxIsLeftBoundary)+1
                maximums = np.maximum(numObs, newObsArr[startIdx:endIdx]) 
                newObsArr[startIdx:endIdx] = maximums;
        assert all([x != -1 for x in newObsArr]), newObsArr;
        assert len(newObsArr)==newObsArrLen;
        return newObsArr;
    
    def addSummedDataTrack(self, key, dataTrack):
        #assert key not in self.summedDataTracks\
        #    , key+" already in summedDataTracks" 
        self.summedDataTracks[key] = dataTrack;
        transformedNumObs=(
            self.transformNumObsAccordingToWidthAndStride(
                self.num_underlying_observations
                , effectiveStride=dataTrack.effectiveStride
                , effectiveWidth=dataTrack.effectiveWidth
                , layerFromAbove=dataTrack.layerFromAbove
                , minVisibility=dataTrack.minVisibility)[None,:])
        assert transformedNumObs.shape[1]==dataTrack.data.shape[1]\
                ,key+" "+str(dataTrack.data.shape)+\
                     " "+str(transformedNumObs.shape)+" "\
                    +str(self.num_underlying_observations.shape)+" "\
                    +str(dataTrack.effectiveStride)\
                    +" "+str(dataTrack.effectiveWidth)
        if (dataTrack.pseudocount is not None):
            (self.normalisedDataTracks[key],
              self.revCompedNormalisedDataTracks[key]) =\
               [(data + self.pseudocountMultiplier*dataTrack.pseudocount)\
                /(numObs+self.pseudocountMultiplier)
                for data,numObs in\
                    [(dataTrack.data, transformedNumObs),
                     (dataTrack.revCompData, transformedNumObs[:,::-1])]] 
        else: 
            (self.normalisedDataTracks[key],
              self.revCompedNormalisedDataTracks[key]) =\
               [(data)/transformedNumObs for data in
                [dataTrack.data, dataTrack.revCompData]] 

    def merge(self, otherModiscoMotif
                  , subtracks_to_align_on
                  , subtrackNormaliseFunc
                  , normaliseFunc
                  , smallerPerPosNormFuncs
                  , largerPerPosNormFuncs
                  , revComp):
        """
            subtracks_to_align_on: subtracks to use for finding optimal
                                alignment
            subtracks_to_align_on, subtrackNormaliseFunc, normaliseFunc
                smallerPerPosNormFuncs, largerPerPosNormFuncs, revComp:
                    you should make these the same as the arguments
                    passed to get_correlation_matrix
            revComp: boolean indicating whether to consider the
                reverse complement as well
        """
        #selfTransformed, otherTransfored are basically the numpy array
        #that will be used for cross correlation (obtained from the
        #appropriate subtracks), with any normalization applied
        selfTransformed, otherTransformed =\
            [getArrayForCrossCorrFromModiscoMotif(
                grammar=grammar
                , subtracks_to_align_on=subtracks_to_align_on
                , subtrackNormaliseFunc=subtrackNormaliseFunc
                , useSummed=False
                , revComp=False)\
             for grammar in [self, otherModiscoMotif]]; 
        if (revComp):
            otherTransformedRevComp = getArrayForCrossCorrFromModiscoMotif(
                            grammar=otherModiscoMotif
                            , subtracks_to_align_on=subtracks_to_align_on 
                            , subtrackNormaliseFunc=subtrackNormaliseFunc
                            , useSummed=False
                            , revComp=True);

        #find the alignment with the optimal overlap
        bestCorrelation, shift, firstIsSmaller =\
            util.getBestLengthwiseCrossCorrelationOfArrays(\
                selfTransformed
                , otherTransformed
                , normaliseFunc=normaliseFunc
                , smallerPerPosNormFuncs=smallerPerPosNormFuncs
                , largerPerPosNormFuncs=largerPerPosNormFuncs)
        useRevComp=False;
        if (revComp):    
            bestCorrelationRevComp, shiftRevComp, firstIsSmallerRevComp =\
                util.getBestLengthwiseCrossCorrelationOfArrays(\
                    selfTransformed
                    , otherTransformedRevComp
                    , normaliseFunc=normaliseFunc
                    , smallerPerPosNormFuncs=smallerPerPosNormFuncs
                    , largerPerPosNormFuncs=largerPerPosNormFuncs)
            assert firstIsSmallerRevComp==firstIsSmaller;
            if (bestCorrelationRevComp > bestCorrelation):
                useRevComp=True;
                shift=shiftRevComp;
                otherModiscoMotif=otherModiscoMotif.getRevCompModiscoMotif();
        if (firstIsSmaller):
            smaller = self;
            larger = otherModiscoMotif;
        else:
            smaller = otherModiscoMotif;
            larger = self;
        #having found the optimal alignment, do merging 
        return self.mergeArraysTogether(smaller=smaller
                                        , larger=larger
                                        , shift=shift);
    @staticmethod
    def obtainLeftPadRightPadLeftIdxRightIdx(smallerLen, largerLen, shift
                                            , effectiveStride, effectiveWidth
                                            , layerFromAbove):
        #effectiveWidth does not actually factor in.
        if (layerFromAbove==True):
            assert effectiveStride==1, "have not dealt with stride>1 for"\
                                       +" layerFromAbove=True"
        assert effectiveStride <= effectiveWidth; #usually catches if you have
                                                  #flipped width & stride
        shift=shift*effectiveStride
        leftPad = max(0, -shift);
        rightPad = max(0, (smallerLen+shift)-largerLen);
        leftIdx = (shift+leftPad);
        rightIdx = (leftIdx+smallerLen);
           
        return (leftPad, rightPad, leftIdx, rightIdx);
        
    def mergeArraysTogether(self, smaller, larger, shift):
        newTotalObservationsEver = larger.totalObservationsEver\
                                    + smaller.totalObservationsEver;
        newNumUnderlyingObservations = self.padAndAdd_1d(
                                        smaller.num_underlying_observations
                                        , larger.num_underlying_observations
                                        , shift
                                        , effectiveStride=1
                                        , effectiveWidth=1
                                        , layerFromAbove=False);
        newSummedDataTracks = {}
        for aKey in larger.summedDataTracks:
            effectiveStride=smaller.summedDataTracks[aKey].effectiveStride
            effectiveWidth=smaller.summedDataTracks[aKey].effectiveWidth
            layerFromAbove=smaller.summedDataTracks[aKey].layerFromAbove
            fillValue=smaller.summedDataTracks[aKey].fillValue
            minVisibility=smaller.summedDataTracks[aKey].minVisibility
            data, revCompData = self.padAndAdd_2d(
                 smallerArray=smaller.summedDataTracks[aKey].data 
               , largerArray=larger.summedDataTracks[aKey].data
               , smallerArrayRevComp=smaller.summedDataTracks[aKey].revCompData
               , largerArrayRevComp=larger.summedDataTracks[aKey].revCompData
               , shift=shift
               , effectiveStride=effectiveStride
               , effectiveWidth=effectiveWidth
               , layerFromAbove=layerFromAbove)
            newSummedDataTracks[aKey] =\
                                     DataTrack(
                                       data=data
                                     , revCompData=revCompData
                                     , pseudocount=\
                                        self.summedDataTracks[aKey].pseudocount
                                     , effectiveStride=effectiveStride
                                     , effectiveWidth=effectiveWidth
                                     , layerFromAbove=layerFromAbove
                                     , fillValue=fillValue
                                     , minVisibility=minVisibility);
        return ModiscoMotif(summedDataTracks=newSummedDataTracks
                       ,num_underlying_observations=newNumUnderlyingObservations
                       ,totalObservationsEver=newTotalObservationsEver
                       ,minPseudocount=self.minPseudocount
                       ,pseudocountFrac=self.pseudocountFrac); 
       
    def padAndAdd_1d(self, smallerArray, largerArray, shift, effectiveStride, effectiveWidth, layerFromAbove):
        assert len(smallerArray.shape)==1;
        assert len(largerArray.shape)==1;
        (leftPad, rightPad, leftIdx, rightIdx) = self.obtainLeftPadRightPadLeftIdxRightIdx(len(smallerArray), len(largerArray)
                                                            , shift, effectiveStride, effectiveWidth, layerFromAbove)
        newArray = np.pad(largerArray, pad_width=[(leftPad, rightPad)], mode='constant');
        newArray[leftIdx:rightIdx] += smallerArray;
        return newArray;
    
    def padAndAdd_2d(self, smallerArray, largerArray,
                           smallerArrayRevComp, largerArrayRevComp,
                           shift, effectiveStride,
                           effectiveWidth, layerFromAbove):
        assert len(smallerArray.shape)==2;
        assert len(largerArray.shape)==2;
        (leftPad, rightPad, leftIdx, rightIdx) =\
          self.obtainLeftPadRightPadLeftIdxRightIdx(
           smallerArray.shape[1], largerArray.shape[1]
           , shift, effectiveStride, effectiveWidth, layerFromAbove)

        newArray = np.pad(largerArray, pad_width=[
                                        (0,0)
                                        , (leftPad, rightPad)]
                                        , mode='constant')
        newArray[:,leftIdx:rightIdx] += smallerArray; 
        assert(newArray.shape[1]>=largerArray.shape[1])
        assert(newArray.shape[1]>=smallerArray.shape[1])

        newArrayRevComp = np.pad(largerArrayRevComp, pad_width=[
                                                     (0,0)
                                                     , (rightPad, leftPad)]
                                                     , mode='constant')
        newArrayRevComp[:,
                        (newArray.shape[1]-rightIdx):
                        (newArray.shape[1]-leftIdx)]\
                        += smallerArrayRevComp 

        return newArray, newArrayRevComp

class Seqlet(ModiscoMotif):
    SeqletJsonKeys = util.enum(location="location", sequenceId="sequenceId");
    def __init__(self, location
                     , sequenceId
                     , *args
                     , **kwargs):
        super(type(self), self).__init__(*args, **kwargs);
        self.location=location;
        self.sequenceId=sequenceId;
    def extractDataForSummedDataTrack(self
                                        , key_name
                                        , full_data_arr
                                        , pseudocount
                                        , fullRevCompDataArr
                                        , rev_comp_func
                                        , effectiveStride
                                        , effectiveWidth
                                        , layerFromAbove
                                        , fillValue
                                        , minVisibility):
        """
            layerFromAbove: changes interpretation of stride and width;
                the layer being augmented exists *above* the layer
                that self.location references
            minVisibility: if a layerFromAbove, make sure that the neurons
                in this conv layer that are included can see at least
                minVisibility bases of the layer in question
        """
        assert rev_comp_func is None or fullRevCompDataArr is None,\
            "exactly one of rev_comp_func or fullRevCompDataArr should be None" 
        assert rev_comp_func is not None or fullRevCompDataArr is not None,\
            "exactly one of rev_comp_func or fullRevCompDataArr should be None"
        if (fullRevCompDataArr is not None):
            assert full_data_arr.shape==fullRevCompDataArr.shape,\
            "full_data_arr and fullRevCompDataArr need to be the same shape;"\
            +" are "\
            +str(full_data_arr.shape)+" and "+str(fullRevCompDataArr.shape)
        if (layerFromAbove==False):
            leftIdx = self.location[0]*effectiveStride
            rightIdx = (self.location[1]-1)*effectiveStride + effectiveWidth
            data=full_data_arr[self.sequenceId,:,leftIdx:rightIdx]
            if (fullRevCompDataArr is not None):
                revCompData = fullRevCompDataArr[self.sequenceId,:,
                                          (full_data_arr.shape[-1]-rightIdx):
                                          (full_data_arr.shape[-1]-leftIdx)]
            else:
                revCompData = rev_comp_func(data)
        elif (layerFromAbove==True):
            if (minVisibility is None):
                minVisibility = int(np.ceil(effectiveWidth/2.0))
            assert (self.location[1]-self.location[0]) > minVisibility,\
                "minVisibility is set to "+str(minVisibility)+" but the len"\
                +" of the location is: "+str(self.location[1]-self.location[0])
            assert fillValue is not None,\
             "fillValue must not be None if layerFromAbove=True because"\
             " layers above have fewer dimensions and may require filling"
            assert effectiveStride==1; #only handling stride of 1 for now
            #find idx in conv layer whose right-most edge-of-view can just
            #(see location[0]+minVisibility-1) in prev layer
            leftIdx = (self.location[0]+minVisibility-1)-(effectiveWidth-1) 
            #in conv layer, idx i is the one whose leftmost edge-of-view
            #can see idx i in prev layer. Since we are doing slices,
            #this is the index in the conv layer whose leftmost edge-of-view
            #just misses the segment.
            rightIdx = (self.location[1]-(minVisibility-1)) 
            lengthToExtract=rightIdx-leftIdx;
            numChannels=full_data_arr.shape[1]
            maxLen=full_data_arr.shape[2]
            #note that the computed leftidx and rightidx may be before the
            #start or beyond the end of the conv layer, if there exists
            #no neuron which "just touches" the provided segment. In that
            #situation we just use the padding provided by fillValue as a
            #placeholder for those nonexistent conv neurons
            leftPadding = max(0,0-leftIdx);
            rightPadding = max(0,(rightIdx-maxLen));
            #unpadded length is the length we will actually extract
            #from the data
            unpaddedLength = lengthToExtract-(rightPadding+leftPadding)
            data_unpaddedLeftIdx=leftPadding
            data_unpaddedRightIdx=leftPadding+unpaddedLength
            full_data_arr_leftIdx=max(leftIdx,0)
            full_data_arr_rightIdx=min(maxLen,rightIdx)
            data = np.ones((numChannels, lengthToExtract))*fillValue
            data[:,data_unpaddedLeftIdx:data_unpaddedRightIdx]\
                = full_data_arr[self.sequenceId,:,
                              (full_data_arr_leftIdx):
                              (full_data_arr_rightIdx)];
            if (fullRevCompDataArr is not None):
                revCompData = np.ones((numChannels, lengthToExtract))\
                                                           *fillValue
                revCompData[:, (data.shape[1]-data_unpaddedRightIdx):
                               (data.shape[1]-data_unpaddedLeftIdx)]\
                 = fullRevCompDataArr[
                    self.sequenceId, :,
                    (full_data_arr.shape[-1]-full_data_arr_rightIdx):
                    (full_data_arr.shape[-1]-full_data_arr_leftIdx)]
            else:
                revCompData = rev_comp_func(data)
        else:
            raise RuntimeError("Unsupported val for layerFromAbove: "+str(layerFromAbove));
            
        self.addSummedDataTrack(key_name,\
                                    DataTrack(data
                                      , revCompData=revCompData
                                      , pseudocount=pseudocount
                                      , effectiveStride=effectiveStride
                                      , effectiveWidth=effectiveWidth
                                      , layerFromAbove=layerFromAbove
                                      , fillValue=fillValue
                                      , minVisibility=minVisibility));
    def getJsonableObject(self):
        theClass = self.__class__.__name__
        jsonableObject = super(type(self), self).getJsonableObject();
        jsonableObject[Seqlet.SeqletJsonKeys.location] = self.location;
        jsonableObject[Seqlet.SeqletJsonKeys.sequenceId] = self.sequenceId;
        return jsonableObject;
        
    @classmethod
    def getLoadingKwargsFromJsonableObject(cls, jsonableObject):
        loadingKwargs = ModiscoMotif.getLoadingKwargsFromJsonableObject(jsonableObject);
        loadingKwargs[Seqlet.SeqletJsonKeys.location] = jsonableObject[Seqlet.SeqletJsonKeys.location]
        loadingKwargs[Seqlet.SeqletJsonKeys.sequenceId] = jsonableObject[Seqlet.SeqletJsonKeys.sequenceId]
        return loadingKwargs;

#for each example, find the critical subset of positives that outweights the negatives
def findCriticalSubset(singleExampleContribs, outputBeforeActivation=None\
                        , activation=None, thresholdProb=1.0, includeNeg=True):
    if (outputBeforeActivation is None or activation is None):
        assert outputBeforeActivation is None and activation is None #all or nothing
        assert thresholdProb==1.0 #don't need these args if including everything
    else:
        assert activation=="sigmoid", "Non sigmoid activation not supported yet!"
    assert thresholdProb >= 0.5 and thresholdProb <= 1.0
    ravelledContribs = enumerate(np.ravel(singleExampleContribs));
    summed_negative = 0;
    totalContribsSum = 0;
    ravelled_positive = [];
    ravelled_all = []
    #assert len(ravelledContribs) > 0
    #partition by negative and positive
    for contrib in ravelledContribs:
        totalContribsSum += contrib[1];
        if contrib[1] < 0:
            summed_negative += contrib[1];
        elif (contrib[1] > 0):
            ravelled_positive.append(contrib);
        #mystery bug...using ravelledContribs in place of
        #ravelled_all does not work...somehow in some cases
        #ravelledContribs gets emptied...
        ravelled_all.append(contrib)
    if (outputBeforeActivation is not None):
        netBias = outputBeforeActivation - totalContribsSum
        sumSoFar = summed_negative + netBias;
    criticalSubsetIndices = []
    criticalSubsetContributions = [];
   
    if (outputBeforeActivation is not None): 
        if (activation=="sigmoid"):
            activationFunc = lambda x: 1.0/(1.0 + np.exp(-x)); #sigmoid activation
        else:
            raise RuntimeError("Unsupported activation:",activation);

    sortedPositives = sorted(ravelled_positive, key=lambda x: -x[1]);
    sortedAll = sorted(ravelled_all, key=lambda x: -x[1]);
    assert len(sortedPositives) > 0;
    if (len(sortedAll)==0):
       print("wtf",sortedPositives)
    assert len(sortedAll)>0
    i = 0;
    if (thresholdProb==1.0):
        if (includeNeg==True):
            criticalSubsetIndices.extend(x[0] for x in sortedAll);
            criticalSubsetContributions.extend(x[1] for x in sortedAll);
        else:
            criticalSubsetIndices.extend(x[0] for x in sortedPositives);
            criticalSubsetContributions.extend(x[1] for x in sortedPositives); 
    else: 
        while (activationFunc(sumSoFar) < thresholdProb) and (i < len(sortedPositives)):
        #while (sortedPositives[i][1] >= (0.1)*sortedPositives[0][1] and i < len(sortedPositives)):
            sumSoFar += sortedPositives[i][1];
            criticalSubsetIndices.append(sortedPositives[i][0]);
            criticalSubsetContributions.append(sortedPositives[i][1])
            i += 1
    #convert the ravelled incides to unravelled indices
    if (len(criticalSubsetIndices) == 0):
        print("WARN: found an example which has no positive deepLIFT"
              "contribs and output before activation:",outputBeforeActivation)
        print(summed_negative)
        print(outputBeforeActivation)
        print(totalContribsSum)
        unravelledIndices = []
        assert False
    else:
        unravelledIndices = list(zip(*np.unravel_index(criticalSubsetIndices, singleExampleContribs.shape)));
    return zip(unravelledIndices, criticalSubsetContributions);

def groupContribsByPos(criticalSubset):
    """
        criticalSubset: array with elements of shape:
            ((channel, row, col), contribution)
        at least one of channel or row must have max
            size 1.
        Returns: dictionary of the form:
            pos -> [array of (channel+row, contribution)]
            channel+row because at least one of them will be 0
                    
    """
    #return pos -> (channel/row, contrib)
    contribsGroupedByPos = defaultdict(list);
    for ((channel, row, col),contribution) in criticalSubset:
        assert row==0 or channel==0;
        contribsGroupedByPos[col].append((channel+row, contribution))
    return contribsGroupedByPos;

def getTotalContribAtPoses(positions, contribsGroupedByPos):
    return [sum(x[1] for x in contribsGroupedByPos[pos])
                for pos in positions]; 
def getPosToTotalContrib(contribsGroupedByPos):
    return dict((pos, sum(x[1] for x in contribsGroupedByPos[pos]))
                 for pos in contribsGroupedByPos);

def getRepresentedChannels(criticalSubset):
    return sorted(Set(x[0] for x in criticalSubset));

def findContiguousWithoutGaps(positions, allowedGap):
    """
        positions: sorted array of positions
        allowedGap = None means take whole seq
    """
    lastPos = positions[0];
    segments = [];
    if allowedGap is None:
        segments.append((positions[0], positions[-1]));
        return segments;
    start = lastPos;
    for pos in positions[1:]:
        if pos - lastPos > allowedGap:
            segments.append((start, lastPos));
            start = pos;
        lastPos = pos
    if start!=pos: #handle last position
        segments.append((start,pos));
    return segments;

class AbstractSegmentIdentifier(object):
    def __call__(self, criticalSubset, numCols):
        """
            return continuous segments
        """
        raise NotImplementedError();
    
class FullSegment(AbstractSegmentIdentifier):
    def __call__(self, contribsGroupedByPos, numCols):
        #find the min position
        sortedPositions = sorted(contribsGroupedByPos.keys());
        return [(sortedPositions[0], sortedPositions[-1])]; 

class FixedWindowAroundPeaks(AbstractSegmentIdentifier):
    """
    Algorithm is as follows:
       compute sums of the deepLIFT contributions in sliding window of size sliding_window_for_max_size
       find peaks (points whose sliding window sums are larger than their neighbours; for plateaus, take the middle)
       filter out peaks which are not at least ratio_to_top_peaks_to_include of the tallest peak
       for each peak in order of highest peak first:
          add (peakStart-flank_to_expand_around_peak_size
              , peakStart+sliding_window_for_max_size+flank_to_expand_around_peak_size)
          to your list of identified segments
          filter out any peaks that are within exclude_peaks_within_window of this peak to your list
       loop until there are no more candidate peaks left or the total number of segments identified is max_segments
    """
    def __init__(self, sliding_window_for_max_size
                     , flank_to_expand_around_peak_size 
                     , exclude_peaks_within_window
                     , ratio_to_top_peaks_to_include
                     , max_segments):
        self.sliding_window_for_max_size = sliding_window_for_max_size;
        self.flank_to_expand_around_peak_size = flank_to_expand_around_peak_size
        self.exclude_peaks_within_window = exclude_peaks_within_window;
        self.ratio_to_top_peaks_to_include = ratio_to_top_peaks_to_include;
        self.max_segments = max_segments;

    def __call__(self, contribsGroupedByPos, numCols):
        posToTotalContrib = getPosToTotalContrib(contribsGroupedByPos); 
        return self.getSegments(util.SparseArrFromDict(
                                        theDict=posToTotalContrib
                                        , defaultVal=0
                                        , totalLen=numCols));

    def getSegments(self, arr):
        #compute sum using sliding window
        totalContribsRunningWindowSum = util.computeRunningWindowSum(
                                            arr=arr
                                            ,windowSize=self.sliding_window_for_max_size);
        return self.getSegmentsFromRunningWindowSum(totalContribsRunningWindowSum);

    def getSegmentsFromRunningWindowSum(self, totalContribsRunningWindowSum):
        numCols = len(totalContribsRunningWindowSum)+self.sliding_window_for_max_size-1
        #find peaks
        potentialPeaks = identifyPeaks.identifyPeaks(totalContribsRunningWindowSum);
        if (len(potentialPeaks)==0):
            topLocation = np.argmax(totalContribsRunningWindowSum);
            assert topLocation==0 or topLocation==len(totalContribsRunningWindowSum)-1;
            potentialPeaks = [(topLocation, totalContribsRunningWindowSum[topLocation])];
            maxPeak = potentialPeaks[0][1];
        else:
            maxPeak = max([x[1] for x in potentialPeaks]);
        #filter out all peaks < "ratio" of max peak
        potentialPeaks = [x for x in potentialPeaks 
                            if x[1] 
                            >= self.ratio_to_top_peaks_to_include*maxPeak];
        segments = []
        #find the max peak
        while len(potentialPeaks) > 0 and len(segments) < self.max_segments:
            ((maxPeakIdx, maxPeak), maxPeak) = util.getBest(potentialPeaks
                                                    , lambda x: x[1]
                                                    , takeMax=True);  
            #the running window sum returns the leftmost index. So
            segments.append((max(0,maxPeakIdx-self.flank_to_expand_around_peak_size)
                            , min(maxPeakIdx+self.sliding_window_for_max_size
                                    +self.flank_to_expand_around_peak_size, numCols)));
            #filter out any peaks within self.exclude_peaks_within_window of
            #the peak
            potentialPeaks = [x for x in potentialPeaks
                                if abs(x[0]-maxPeakIdx) > self.exclude_peaks_within_window]
        return segments;

class AbstractKeepLookingFunc(object):
    def __call__(self, potentialNextPos, currentPos
                     , potentialNextContrib, thisPeakContrib
                     , maxContrib):
        raise NotImplementedError();         

#converts a list of dictionaries to a numpy mat
def dictListToNumpyMatrix(dictList, numRows):
    toReturn = np.zeros((numRows, len(dictList)));
    for (posIdx, posDict) in enumerate(dictList):
        for channel in posDict:
            toReturn[int(channel), posIdx] += posDict[channel]
    return toReturn;

#single region!
def get_seqletsForArrayOfContribs(singleExampleContribs
                                  , wideRevCompArray
                                  , rev_comp_func
                                  , outputBeforeActivation
                                  , activation, segment_identifier
                                  , thresholdProb, sequenceId
                                  , includeNeg):
    """
        Returns an array of ModiscoMotif objects for a SINGLE REGION
        (singleExampleContribs is for a single region)
    """
    assert wideRevCompArray is None or rev_comp_func is None,\
     "Exactly one of wideRevCompArray and rev_comp_func should not be None"
    assert (wideRevCompArray is not None) or (rev_comp_func is not None),\
     "Exactly one of wideRevCompArray and rev_comp_func should not be None"

    contribsInKeySegments, keySegments =\
        extractKeySegments(singleExampleContribs, outputBeforeActivation
                           , activation, segment_identifier
                           , thresholdProb, includeNeg)
    numRows = singleExampleContribs.shape[0]+singleExampleContribs.shape[1]-1
    seqlets = []
    for contribs, keySegment in zip(contribsInKeySegments, keySegments):
        data=dictListToNumpyMatrix(contribs, numRows)
        if wideRevCompArray is not None:
            revCompData = wideRevCompArray[:,
                           (wideRevCompArray.shape[-1]-keySegment[1]):
                           (wideRevCompArray.shape[-1]-keySegment[0])]
        else:
            revCompData = rev_comp_func(data)
        seqlets.append(Seqlet(
                        summedDataTracks={
                         ModiscoMotif.core_importance_track:\
                          DataTrack(
                           data=data
                            ,revCompData=revCompData
                            ,pseudocount=0
                            ,effectiveStride=1
                            ,effectiveWidth=1
                            ,layerFromAbove=False
                            ,fillValue=None
                            ,minVisibility=None)}
                        ,num_underlying_observations=1
                        ,totalObservationsEver=1
                        ,location=keySegment
                        ,sequenceId=sequenceId))
    return seqlets;

def extractKeySegments(singleExampleContribs
                       , outputBeforeActivation
                       , activation, segment_identifier
                       , thresholdProb, includeNeg):
    """
        segment_identifier: instance of AbstractSegmentIdentifier; the
            rule for identifying key segments
        numCols: length of the underlying region
    """
    assert activation is None or util.assertIsType(activation, str, "activation");
    criticalSubset = findCriticalSubset(singleExampleContribs=singleExampleContribs
                                        ,outputBeforeActivation=outputBeforeActivation
                                        ,activation=activation
                                        ,thresholdProb=thresholdProb
                                        ,includeNeg=includeNeg) 
    numCols = singleExampleContribs.shape[2];
    assert singleExampleContribs.shape[0]==1 or singleExampleContribs.shape[1]==1 #either channels or rows must have dim 1
    #util.assertIsType(segment_identifier, AbstractSegmentIdentifier, "segment_identifier");
    contribsGroupedByPos = groupContribsByPos(criticalSubset);
    keySegments = segment_identifier(contribsGroupedByPos, numCols);
    #each of the things in key segment seqlets is an array of dicts
    #and the indices of the dicts are supposed to be the fitlers
    contribsInKeySegments = []
    for keySegment in keySegments:
        contribsInKeySegment = [{} for i in range(keySegment[1]-keySegment[0])];
        for (idx,pos) in enumerate(xrange(keySegment[0],keySegment[1])):
            if pos in contribsGroupedByPos:
                for (aFilter, contribution) in contribsGroupedByPos[pos]:
                    assert aFilter not in contribsInKeySegment[idx]
                    contribsInKeySegment[idx][aFilter] = contribution;
        contribsInKeySegments.append(contribsInKeySegment);
    #sort the keySegments by the most important first
    tuplesToSplitUp = sorted(zip(contribsInKeySegments, keySegments)
                    , key=lambda x: -sum([contrib for pos in x[0] for contrib in pos.values()]))
    #split them up
    contribsInKeySegments = [x[0] for x in tuplesToSplitUp]  
    keySegments = [x[1] for x in tuplesToSplitUp]
    return contribsInKeySegments, keySegments;

def getTruePositiveIndicesAboveThreshold(*args, **kwargs):
    print("Deprecated; use deepLIFTutils.getTruePositiveIndicesAboveThreshold");
    return deepLIFTutils.getTruePositiveIndicesAboveThreshold(*args, **kwargs);

###################################################
#I have to endure this nonsense because the function
#pickled by Pool.map has to be accessible at the top level
_seqletsForIdx_singleExampleContribs = util.VariableWrapper(None);
_seqletsForIdx_fullRevCompDataArr = util.VariableWrapper(None)
_seqletsForIdx_rev_comp_func = util.VariableWrapper(None);
_seqletsForIdx_outputsBeforeActivation = util.VariableWrapper(None);
_seqletsForIdx_activation = util.VariableWrapper(None);
_seqletsForIdx_segment_identifier = util.VariableWrapper(None);
_seqletsForIdx_thresholdProb = util.VariableWrapper(None);
_seqletsForIdx_includeNeg=util.VariableWrapper(None);
#Nonsense endured
####################################################
def computeSeqletsForIdx(idx):
    assert _seqletsForIdx_singleExampleContribs.var is not None
    seqlets = get_seqletsForArrayOfContribs(
                    singleExampleContribs=_seqletsForIdx_singleExampleContribs.var[idx]
                    ,wideRevCompArray=(
                       None if _seqletsForIdx_fullRevCompDataArr.var is None else\
                       _seqletsForIdx_fullRevCompDataArr.var[idx])
                    ,rev_comp_func=_seqletsForIdx_rev_comp_func.var
                    , outputBeforeActivation=\
                        None if _seqletsForIdx_outputsBeforeActivation.var is None\
                            else _seqletsForIdx_outputsBeforeActivation.var[idx]
                    , activation=_seqletsForIdx_activation.var
                    , segment_identifier=_seqletsForIdx_segment_identifier.var
                    , thresholdProb=_seqletsForIdx_thresholdProb.var
                    , includeNeg=_seqletsForIdx_includeNeg.var
                    , sequenceId=idx)
    return (seqlets, [idx]*len(seqlets))

def getModiscoMotifs(raw_importance_contribs
                , indicesToGetModiscoMotifsOn
                , outputsBeforeActivation
                , activation
                , thresholdProb
                , segment_identifier
                , **kwargs):
    print("Get grammars is deprecated as the name was confusing; use get_seqlets")
    return get_seqlets(raw_importance_contribs=raw_importance_contribs
                      , indicesToGetSeqletsOn=indicesToGetModiscoMotifsOn
                      , outputsBeforeActivation=outputsBeforeActivation
                      , activation=activation
                      , thresholdProb=thresholdProb
                      , segment_identifier=segment_identifier
                      , **kwargs);

def get_seqlets(raw_importance_contribs
                , segment_identifier
                , fullRevCompDataArr=None
                , rev_comp_func=None
                , includeNeg=True
                , indicesToGetSeqletsOn=None
                , outputsBeforeActivation=None
                , activation=None
                , thresholdProb=1.0
                , numThreads=1
                , secondsBetweenUpdates=1):
    if (rev_comp_func is None and fullRevCompDataArr is None):
        rev_comp_func=RevCompWithDNArowsSubset(dnaRowsStart=0, dnaRowsEnd=4);
        print("No reverse comp function or rev comp array provided"
              "so assuming you have dna as first 4 rows");
    if (indicesToGetSeqletsOn is None):
        indicesToGetSeqletsOn = xrange(len(raw_importance_contribs));
    assert outputsBeforeActivation is None or\
            (len(raw_importance_contribs)==len(outputsBeforeActivation))\
            , "raw_importance_contribs and outputsBeforeActivation should be the same length"\
              "but are "+str(raw_importance_contribs.shape)+" and "+str(outputsBeforeActivation.shape)
    reload(util)
    assert activation is None or util.assertIsType(activation, str, "activation");
    assert len(raw_importance_contribs.shape)==4; #example, channel, rows, cols
    util.assertIsType(thresholdProb, float, "thresholdProb");
    #util.assertIsType(segment_identifier, AbstractSegmentIdentifier, "segment_identifier");
    _seqletsForIdx_singleExampleContribs.var=raw_importance_contribs;
    _seqletsForIdx_fullRevCompDataArr.var=fullRevCompDataArr
    _seqletsForIdx_rev_comp_func.var=rev_comp_func;
    _seqletsForIdx_outputsBeforeActivation.var=outputsBeforeActivation;
    _seqletsForIdx_activation.var=activation;
    _seqletsForIdx_segment_identifier.var=segment_identifier;
    _seqletsForIdx_thresholdProb.var=thresholdProb;
    _seqletsForIdx_includeNeg.var=includeNeg
    if (numThreads > 1):
        seqletsAndIndicesTuples = util.multiprocessing_map_printProgress(
                                    secondsBetweenUpdates=secondsBetweenUpdates
                                    ,numThreads=numThreads
                                    ,func=computeSeqletsForIdx
                                    ,iterable=indicesToGetSeqletsOn);
    else:
        seqletsAndIndicesTuples=[];
        for x in indicesToGetSeqletsOn:
            if (x%100==0):
                print("Done",x,"of",len(indicesToGetSeqletsOn));
            seqletsAndIndicesTuples.append(computeSeqletsForIdx(x));
    #disentangle/unlist seqletsAndIndicesTuples
    seqletsOnAllExamples = []
    indicesOfSeqlets = []
    for seqletsAndIndicesTuple in seqletsAndIndicesTuples:
        seqletsOnAllExamples.extend(seqletsAndIndicesTuple[0]);
        indicesOfSeqlets.extend(seqletsAndIndicesTuple[1]);
    assert len(seqletsOnAllExamples) == len(indicesOfSeqlets);
    for (seqlet, index) in zip(seqletsOnAllExamples, indicesOfSeqlets):
        assert seqlet.sequenceId==index;
    #sort them by highest contributing seqlets
    contribsForSeqlets = [np.sum(seqlet.summedCoreDeepLIFTtrack) for seqlet in seqletsOnAllExamples];
    sortOrder = [x[0] for x in sorted(enumerate(contribsForSeqlets), key=lambda x: -x[1])]; 
    seqletsOnAllExamples = [seqletsOnAllExamples[i] for i in sortOrder];
    indicesOfSeqlets = [indicesOfSeqlets[i] for i in sortOrder];
    return seqletsOnAllExamples, indicesOfSeqlets


###################################################
#I have to endure this nonsense because the function
#pickled by Pool.map has to be accessible at the top level
_computeBestCorrelation_arrays = util.VariableWrapper(None);
_computeBestCorrelation_revCompArrays = util.VariableWrapper(None);
_computeBestCorrelation_account_for_rev_comp = util.VariableWrapper(None);
_computeBestCorrelation_normaliseFunc = util.VariableWrapper(None);
_computeBestCorrelation_smallerPerPosNormFuncs = util.VariableWrapper(None);
_computeBestCorrelation_largerPerPosNormFuncs = util.VariableWrapper(None);
#Nonsense endured
####################################################
def computeBestCorrelation(tupleToCorrelate):
    bestCorrelation, shift, firstIsSmaller =\
        util.getBestLengthwiseCrossCorrelationOfArrays(
            _computeBestCorrelation_arrays.var[tupleToCorrelate[0]]
            , _computeBestCorrelation_arrays.var[tupleToCorrelate[1]]
            , normaliseFunc=_computeBestCorrelation_normaliseFunc.var
            , smallerPerPosNormFuncs=_computeBestCorrelation_smallerPerPosNormFuncs.var
            , largerPerPosNormFuncs=_computeBestCorrelation_largerPerPosNormFuncs.var)
    if (_computeBestCorrelation_account_for_rev_comp.var ==True):
        bestCorrelationRev, shiftRev, firstIsSmallerRev =\
            util.getBestLengthwiseCrossCorrelationOfArrays(
                _computeBestCorrelation_arrays.var[tupleToCorrelate[0]]
                , _computeBestCorrelation_revCompArrays.var[tupleToCorrelate[1]]
                , normaliseFunc=_computeBestCorrelation_normaliseFunc.var
                , smallerPerPosNormFuncs=_computeBestCorrelation_smallerPerPosNormFuncs.var
                , largerPerPosNormFuncs=_computeBestCorrelation_largerPerPosNormFuncs.var)
        return max(bestCorrelation, bestCorrelationRev);
    else:
        return bestCorrelation;

def getArrayForCrossCorrFromModiscoMotif(grammar
                                    , subtracks_to_align_on
                                    , subtrackNormaliseFunc
                                    , useSummed
                                    , revComp):
    arr = np.concatenate([subtrackNormaliseFunc(
                                (grammar.getNormalisedDataTrack(subtrackName)
                                if (not revComp) else
                                grammar.getRevCompedNormalisedDataTrack(subtrackName))
                              if (not useSummed) else
                                (grammar.get_summed_data_track(subtrackName)
                                if (not revComp) else
                                grammar.getRevCompedSummedDataTrack(subtrackName)))
                                for subtrackName in subtracks_to_align_on]
                         , axis=0);
    return arr;

def oneOverLen(arr):
    return arr/float(np.shape(arr)[1])

def getSummedChannelSignals(seqlets, subtracks_to_align_on, revComp):
    #not casting to an array immediately as seqlets could be of varying lengths
    listOfChannelTracks = [getArrayForCrossCorrFromModiscoMotif(seqlet
                                      , subtracks_to_align_on
                                      , subtrackNormaliseFunc=identity
                                      , revComp=revComp
                                      , useSummed=False)
                          for seqlet in seqlets] 
    #sum them along the length axis, which may be of varying lengths
    summedAlongLen = np.array([x.sum(axis=1) for x in listOfChannelTracks])
    return summedAlongLen

def getChannelSignals(seqlets, subtracks_to_align_on, revComp):
    channelSignals = getSummedChannelSignals(
                      seqlets, subtracks_to_align_on, revComp)
    return channelSignals

def euclideanSimilarity(twoDVecs1, twoDVecs2): 
    return -np.linalg.norm(twoDVecs1[:,None,:]\
                            - twoDVecs2[None,:,:], axis=2)

def normaliseTwoDVecsByMagnitude(twoDVecs): 
    norms = np.linalg.norm(twoDVecs,axis=1)[:,None]
    norms = np.maximum(norms,0.0000001)
    return twoDVecs/norms

def cosineSimilarity(twoDVecs1, twoDVecs2):
    #normalise the vectors
    normalisedTwoDVecs1 = normaliseTwoDVecsByMagnitude(twoDVecs1) 
    normalisedTwoDVecs2 = normaliseTwoDVecsByMagnitude(twoDVecs2)
    return np.sum(normalisedTwoDVecs1[:,None,:]\
                   * normalisedTwoDVecs2[None,:,:], axis=2) 

def cosineOnPosSimilarity(twoDVecs1, twoDVecs2):
    twoDVecs1 = twoDVecs1*(twoDVecs1 > 0.0)
    twoDVecs2 = twoDVecs2*(twoDVecs2 > 0.0)
    return cosineSimilarity(twoDVecs1, twoDVecs2)

ChannelSimilarityMode = util.enum(cosine=cosineSimilarity,
                                  cosineOnPos=cosineOnPosSimilarity,
                                  euclidean=euclideanSimilarity)
def getChannelSimilarityMatrix(seqlets,
                               subtracks_to_align_on,
                               channelSimilarityFunc,
                               useRevComp): 
    """
        subtracks_to_align_on refers to subtracts corresponding to a conv
            layer, and the distance matrix will be based on the distance
            between the sum over the conv filters.
    """
    channelSignals = getChannelSignals(seqlets,
                                       subtracks_to_align_on,
                                       revComp=False)
    #channelSignalsRevComp will be the same as channelSignals if
    #useRevComp is False
    channelSignalsRevComp = getChannelSignals(seqlets,
                                              subtracks_to_align_on,
                                              revComp=useRevComp)
    similarity_fwd = channelSimilarityFunc(
                      twoDVecs1=channelSignals,
                      twoDVecs2=channelSignals) 
    similarity_rev = channelSimilarityFunc(
                      twoDVecs1=channelSignals,
                      twoDVecs2=channelSignalsRevComp)
    similarity = np.minimum(similarity_fwd, similarity_rev)
    return similarity

def get_correlation_matrix(seqlets
                         , subtracks_to_align_on=[ModiscoMotif.core_importance_track]
                         , subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm
                         , normaliseFunc=util.CROSSC_NORMFUNC.none
                         , smallerPerPosNormFuncs=[]
                         , largerPerPosNormFuncs=[]
                         , account_for_rev_comp=True
                         , numThreads=1
                         , secondsBetweenUpdates=1
                         , batch_size=None
                         , verbose=True):
    return get_correlation_matrix_diffRowsAndCols(
        rowMotifs=seqlets,
        colMotifs=seqlets,
        subtracks_to_align_on=subtracks_to_align_on,
        subtrackNormaliseFunc=subtrackNormaliseFunc,
        normaliseFunc=normaliseFunc,
        smallerPerPosNormFuncs=smallerPerPosNormFuncs,
        largerPerPosNormFuncs=largerPerPosNormFuncs,
        account_for_rev_comp=account_for_rev_comp,
        numThreads=numThreads,
        secondsBetweenUpdates=secondsBetweenUpdates,
        batch_size=batch_size, 
        verbose=verbose)

def padArraysToBeSameLength(arrays):
    maxLen = np.max([x.shape[-1] for x in arrays])
    arrays_sameLen = []
    for array in arrays:
        if array.shape[-1] < maxLen:
            left_pad = np.ceil(0.5*(maxLen-array.shape[-1]))
            right_pad = np.floor(0.5*(maxLen-array.shape[-1]))
            new_array = np.zeros((array.shape[0], maxLen))
            new_array[:,left_pad:(maxLen-right_pad)] = array
            array=new_array
        arrays_sameLen.append(array)
    return arrays_sameLen

def get_correlation_matrix_diffRowsAndCols(
                         rowMotifs
                         , colMotifs
                         , subtracks_to_align_on
                         , subtrackNormaliseFunc
                         , normaliseFunc
                         , smallerPerPosNormFuncs
                         , largerPerPosNormFuncs
                         , account_for_rev_comp
                         , numThreads
                         , secondsBetweenUpdates
                         , batch_size
                         , verbose):
    #rowMotifs will be padded to be same length and used as filters in GPU conv
    #colMotifs are the ones that will be revComp'd if necessary
    assert batch_size is not None,\
     "I've removed support for non-GPU "\
     +"cross-correlation as it was getting cumbersome to maintain."
    assert len(smallerPerPosNormFuncs)==0, "per pos norm no longer supported as it didn't seem to work well"
    assert len(largerPerPosNormFuncs)==0, "per pos norm no longer supported as it didn't seem to work well"

    startTime = time.time()
    if (verbose):
        print("Num words:",len(rowMotifs),"and",len(colMotifs));
    #if the num of underlying observations is not 1, then
    #should use some kind of normalised arra below and not
    #get_summed_data_track.

    rowMotifs_toCorrelate =[getArrayForCrossCorrFromModiscoMotif(motif
                                              , subtracks_to_align_on
                                              , subtrackNormaliseFunc
                                              , revComp=False
                                              , useSummed=False)
                                  for motif in rowMotifs] 

    colMotifs_toCorrelate =[getArrayForCrossCorrFromModiscoMotif(motif
                                              , subtracks_to_align_on
                                              , subtrackNormaliseFunc
                                              , revComp=False
                                              , useSummed=False)
                                  for motif in colMotifs] 

    #we have to call normaliseFunc on each one
    (rowMotifs_toCorrelate,
     colMotifs_toCorrelate) = [
      padArraysToBeSameLength([normaliseFunc(x) for x in motifs_toCorrelate])
      for motifs_toCorrelate in [rowMotifs_toCorrelate, colMotifs_toCorrelate]]

    rowMotifs_toCorrelate = np.array(rowMotifs_toCorrelate)

    #rev comp the colMotifs if necessary
    if (account_for_rev_comp):
        revComp_colMotifs_toCorrelate =\
            [getArrayForCrossCorrFromModiscoMotif(
              motif
              , subtracks_to_align_on
              , subtrackNormaliseFunc
              , useSummed=False
              , revComp=True) for motif in colMotifs]
        revComp_colMotifs_toCorrelate =\
         padArraysToBeSameLength(
          [normaliseFunc(x) for x in revComp_colMotifs_toCorrelate])

    correlationsNoRevComp = get_max_cross_corr(
                    filters=rowMotifs_toCorrelate.copy(),
                    things_to_scan=[x.copy() for x in colMotifs_toCorrelate],
                    verbose=verbose,
                    batch_size=batch_size)
    if (account_for_rev_comp):
        print("Repeating for reverse complement")
        correlationsRevComp = get_max_cross_corr(
           filters=rowMotifs_toCorrelate.copy(),
           things_to_scan=[x.copy() for x in revComp_colMotifs_toCorrelate],
           verbose=verbose,
           batch_size=batch_size)
        correlations = np.maximum(correlationsNoRevComp,
                                  correlationsRevComp)
    else:
        correlations = correlationsNoRevComp 
    print("Seconds to compute corr mat:",time.time()-startTime); 
    return correlations


def get_conv_out_symbolic_var(input_var,
                              set_of_2d_patterns_to_conv_with,
                              normalise_by_magnitude,
                              take_max):
    assert len(set_of_2d_patterns_to_conv_with.shape)==3
    if (normalise_by_magnitude):
        set_of_2d_patterns_to_conv_with =\
         set_of_2d_patterns_to_conv_with/\
          (np.sqrt(np.sum(np.sum(np.square(set_of_2d_patterns_to_conv_with),
                               axis=-1),
                        axis=-1))[:,None,None])
    filters = theano.tensor.as_tensor_variable(
               x=set_of_2d_patterns_to_conv_with,
               name="filters")
    conv_out = theano.tensor.signal.conv.conv2d(
                input=input_var,
                filters=filters)
    if (normalise_by_magnitude):
        sum_squares_per_pos =\
                   theano.tensor.signal.conv.conv2d(
                    input=theano.tensor.square(input_var),
                    filters=np.ones(set_of_2d_patterns_to_conv_with.shape)\
                            .astype("float32")) 
        per_pos_magnitude = theano.tensor.sqrt(sum_squares_per_pos)
        per_pos_magnitude += 0.0000001*(per_pos_magnitude < 0.0000001)
        conv_out = conv_out/per_pos_magnitude
    if (take_max):
        conv_out = theano.tensor.max(
                    theano.tensor.max(conv_out, axis=-1), #max over cols
                    axis=-1) #max over rows
    return conv_out 


def compile_conv_func_with_theano(set_of_2d_patterns_to_conv_with,
                                  normalise_by_magnitude=False,
                                  take_max=False):
    input_var = theano.tensor.TensorType(dtype=theano.config.floatX,
                                         broadcastable=[False]*3)("input")
    conv_out = get_conv_out_symbolic_var(input_var,
                                 set_of_2d_patterns_to_conv_with,
                                 normalise_by_magnitude=normalise_by_magnitude,
                                 take_max=take_max)
    func = theano.function([input_var],
                           conv_out,
                           allow_input_downcast=True)
    return func 

def get_max_cross_corr(filters, things_to_scan,
                           verbose=True, batch_size=10,
                           func_params_size=1000000,
                           progress_update=1000,
                           min_overlap=0.3):
    """
        func_params_size: when compiling functions
    """
    #reverse the patterns as the func is a conv not a cross corr
    filters = filters.astype("float32")[:,::-1,::-1]
    to_return = np.zeros((filters.shape[0], len(things_to_scan)))
    #compile the number of filters that result in a function with
    #params equal to func_params_size 
    params_per_filter = np.prod(filters[0].shape)
    filter_batch_size = int(func_params_size/params_per_filter)
    filter_length = filters.shape[-1]
    filter_idx = 0 
    while filter_idx < filters.shape[0]:
        if (verbose):
            print("On filters",filter_idx,"to",
                  min(len(filters),(filter_idx+filter_batch_size)))
        filter_batch = filters[filter_idx:(filter_idx+filter_batch_size)]
        cross_corr_func = compile_conv_func_with_theano(
                           set_of_2d_patterns_to_conv_with=filter_batch,
                           normalise_by_magnitude=False,
                           take_max=True)  
        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = [np.pad(array=x,
                              pad_width=((padding_amount, padding_amount)),
                              mode="constant") for x in things_to_scan]
        max_cross_corrs = np.array(deeplift.util.run_function_in_batches(
                            func=cross_corr_func,
                            input_data_list=[padded_input],
                            batch_size=batch_size,
                            progress_update=(None if verbose==False else
                                             progress_update)))
        assert len(max_cross_corrs.shape)==2, max_cross_corrs.shape
        to_return[filter_idx:
                  (filter_idx+filter_batch_size),:] =\
                  np.transpose(max_cross_corrs)
        filter_idx += filter_batch_size
        
    return to_return



def augmentModiscoMotifsWithData(grammars, *args, **kwargs):
    print("Deprecated; use augment_seqlets_with_data");
    return augment_seqlets_with_data(*args, seqlets=grammars, **kwargs);
def augment_seqlets_with_data(seqlets, full_data_arr
                            , key_name
                            , pseudocount
                            , rev_comp_func=None
                            , fullRevCompDataArr=None
                            , indicesToSubset=None
                            , effectiveStride=1
                            , effectiveWidth=1
                            , layerFromAbove=False
                            , fillValue=None
                            , minVisibility=None):
    assert rev_comp_func is None or fullRevCompDataArr is None,\
     "Exactly one of rev_comp_func or fullRevCompDataArr should be None"
    assert rev_comp_func is not None or fullRevCompDataArr is not None,\
     "Exactly one of rev_comp_func or fullRevCompDataArr should be None"
    if (indicesToSubset is not None):
        full_data_arr = [full_data_arr[i] for i in indicesToSubset];
    for seqlet in seqlets:
        seqlet.extractDataForSummedDataTrack(
                key_name=key_name
                , full_data_arr=full_data_arr
                , pseudocount=pseudocount
                , fullRevCompDataArr=fullRevCompDataArr
                , rev_comp_func=rev_comp_func
                , effectiveStride=effectiveStride
                , effectiveWidth=effectiveWidth
                , layerFromAbove=layerFromAbove
                , fillValue=fillValue
                , minVisibility=minVisibility); 

#create a merged grammar for the clusters
def create_merged_modisco_motifs(clusterLabels, grammars
                         , subtracks_to_align_on=[ModiscoMotif.core_importance_track]
                         , subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm
                         , normaliseFunc=util.CROSSC_NORMFUNC.none
                         , smallerPerPosNormFuncs=[]
                         , largerPerPosNormFuncs=[] 
                         , account_for_rev_comp=True):
    clusterLabelToMergedModiscoMotif = {};
    for clusterLabel, grammar in zip(clusterLabels, grammars):
        if clusterLabel not in clusterLabelToMergedModiscoMotif:
            clusterLabelToMergedModiscoMotif[clusterLabel] = grammar;
        else:
            clusterLabelToMergedModiscoMotif[clusterLabel] = clusterLabelToMergedModiscoMotif[clusterLabel]\
                                                        .merge(grammar
                                                               , subtracks_to_align_on=subtracks_to_align_on
                                                               , subtrackNormaliseFunc=subtrackNormaliseFunc
                                                               , normaliseFunc=normaliseFunc
                                                               , smallerPerPosNormFuncs=smallerPerPosNormFuncs
                                                               , largerPerPosNormFuncs=largerPerPosNormFuncs
                                                               , revComp=account_for_rev_comp)
    return clusterLabelToMergedModiscoMotif

def get_tsne_embedding_of_modisco_motifs(grammarsCorrMat, perplexity, verbose=0, random_state=None):
    import sklearn;
    from sklearn import manifold;
    tsne = manifold.TSNE(metric='precomputed', perplexity=perplexity
                         , verbose=verbose, random_state=random_state);
    grammarsDistMat = np.max(grammarsCorrMat)-grammarsCorrMat
    embedding = tsne.fit_transform(grammarsDistMat)
    return embedding;

def colorTSNEembeddingBySpectralClustering(mat, embedding, n_clusters, colors=None
                                                          , affinity='precomputed'
                                                          , *args, **kwargs):
    if (n_clusters==1):
        labels = [0 for x in embedding];
    else:
        labels = getSpectralClustering(mat, n_clusters, affinity);
    scatter_plot(embedding, labels=labels, colors=colors, *args, **kwargs); 
    return labels;

def colorTSNEembeddingByClusterer(embedding, clusterer, colors=None, *args, **kwargs):
    labels = clusterer.fit_predict(embedding);
    scatter_plot(embedding, labels=labels, colors=colors, *args, **kwargs); 
    return labels;

def getSpectralClustering(mat, n_clusters, affinity):
    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    labels = spectral.fit_predict(mat)
    return labels;

def getKMeansClustering(mat, **kwargs):
    import sklearn.cluster
    clf = sklearn.cluster.KMeans(**kwargs)
    labels = clf.fit_predict(mat)
    return labels;

def getRunningSumOfPositionWeights(array, startFromLeft):
    theLen = array.shape[1];
    runningSum = np.zeros(array.shape[1]+1) #+1 because we want to have the value for an inclusive end
    sumSoFar = 0;
    for i in xrange(theLen):
        if (startFromLeft):
            idx = i;
        else:
            idx = theLen-i;
        if (i > 0): #aha this fixes all my problems
            sumSoFar += np.sum(array[:,idx-1 if startFromLeft else idx]);
        runningSum[idx] = sumSoFar;
    return runningSum 

def adjustModiscoMotifUsingTrimmingCriterion(grammar, trimming_func):
    (start, end) = trimming_func(grammar);
    summedDataTracks = {}
    for key,dataTrack in grammar.summedDataTracks.items():
        if (dataTrack.layerFromAbove==False):
            start_idx=start*dataTrack.effectiveStride
            #(end-1) because "end" is the boundary i.e. it has +1 added
            end_idx=(((end-1)*dataTrack.effectiveStride)
                     +dataTrack.effectiveWidth)
        elif (dataTrack.layerFromAbove):
            assert dataTrack.effectiveStride==1,\
             "Have not handled stride != 1 yet for layerFromAbove"
            start_idx = start
            end_idx = dataTrack.data.shape[-1]-\
                      (grammar.num_underlying_observations.shape[-1]-end)
        data=dataTrack.data[:, start_idx:end_idx]
        revCompData=dataTrack.revCompData[:,
                                          (dataTrack.data.shape[-1]-end_idx):
                                          (dataTrack.data.shape[-1]-start_idx)]
        summedDataTracks[key] = DataTrack(
                                 data=data
                                 , revCompData=revCompData
                                 , pseudocount=dataTrack.pseudocount
                                 , effectiveStride=dataTrack.effectiveStride
                                 , effectiveWidth=dataTrack.effectiveWidth
                                 , layerFromAbove=dataTrack.layerFromAbove
                                 , fillValue=dataTrack.fillValue
                                 , minVisibility=dataTrack.minVisibility) 
    return ModiscoMotif(num_underlying_observations=\
                grammar.num_underlying_observations[start:end]
                ,totalObservationsEver=grammar.totalObservationsEver
                ,summedDataTracks=summedDataTracks 
                ,minPseudocount=grammar.minPseudocount
                ,pseudocountFrac=grammar.pseudocountFrac)

def adjust_modisco_motifs_using_trimming_criterion(labelToModiscoMotif, trimming_func):
    """
        labelsToModiscoMotif is a dictionary, indented to be the dict
            produced by create_merged_modisco_motifs
    """
    toReturn = {}
    for (label, grammar) in labelToModiscoMotif.items():
        toReturn[label] =  adjustModiscoMotifUsingTrimmingCriterion(grammar, trimming_func)
    return toReturn;

class TrimmingFunc(object):
    def __call__(self, grammar):
        raise NotImplementedError();

class TrimArrayColumnsToNumUnderlyingObs(TrimmingFunc):
    def __init__(self, percentObs):
        self.percentObs=percentObs;
    def __call__(self, grammar):
        """
            Will retain all indices where num_underlying_observations
                is at least percentObs of totalObservationsEver
        """
        filteredIndices = [x[0] for x in enumerate(grammar.num_underlying_observations)\
                              if x[1] >= self.percentObs*grammar.totalObservationsEver]  
        if (len(filteredIndices)==0):
            raise RuntimeError(
                "Trimming critierion was such that all positions"
                +" got trimmed; num underlying obs was: "
                +str(grammar.num_underlying_observations)
                +", total observations ever was "
                +str(grammar.totalObservationsEver)
                +" and percent was "+str(self.percentObs))
        return (filteredIndices[0], filteredIndices[-1]+1);

class TrimArrayColumnsToPercent(TrimmingFunc):
    def __init__(self, percent):
        self.percent = percent;
        print("WARNING: this function has not been updated")
    def __call__(self, grammar):
        """
            Will find the smallest subset of the array that retains x% of the signal.
        """
        array = grammar.summedCoreDeepLIFTtrack
        #for now implement brute force thing because not using it for anything intensive but
        #i am pretty sure this can be done more efficiently.
        assert np.sum(np.abs(array)-array) == 0;
        totalSum = np.sum(array);
        #compute running sums of what would be excluded from the left and the right
        sumsFromLeft = getRunningSumOfPositionWeights(array, startFromLeft=True);
        sumsFromRight = getRunningSumOfPositionWeights(array, startFromLeft=False);
        bestTrim = util.GetBest_Min()
        #try all combos of start and end.
        for (leftEdge, sumFromLeft) in enumerate(sumsFromLeft):
            for (rightEdge, sumFromRight) in enumerate(sumsFromRight):
                if (rightEdge > leftEdge):
                    if (totalSum - (sumFromLeft+sumFromRight) >= totalSum*self.percent):
                        bestTrim.process((leftEdge, rightEdge), (rightEdge-leftEdge));
        (bestLeft, bestRight) = bestTrim.getBestObj();
        return array[:,bestLeft:bestRight], (bestLeft, bestRight);

class TrimArrayColumnsToPeak(TrimmingFunc):
    def __init__(self, slidingWindowSizeForPeak, flanksToExpand, trackNameToUse, useRangeNotSum):
        """
            Will look at the summed version of trackNameToUse (so as to overweight positions
                with more observations). Will find the sliding window of size
                slidingWindowSizeForPeak of the highest weight, and will expand by
                flanksToExpand on either side. useRangeNotSum=True will use the range between
                the values(bases) at each position as the weight (appropriate for, eg, gradients
                on sequence).
            Recommended settings for sequence data:
                trackNameToUse = [name of gradients track, usually "gradients"]
                useRangeNotSum = True
            Recommended settings for all other data:
                trackNameToUse = ModiscoMotif.core_importance_track
                useRangeNotSum = False
        """ 
        self.slidingWindowSizeForPeak = slidingWindowSizeForPeak;
        self.flanksToExpand = flanksToExpand; 
        self.trackNameToUse = trackNameToUse;
        self.useRangeNotSum=useRangeNotSum;
    def __call__(self, grammar): 
        """
            Using a sliding window of size slidingWindowSizeForPeak,
                will find the peak in the importance of array cols.
                Will then expand
                the sliding window by flanksToExpand, and return that
                as the final array.
        """
        #NOTE the use of the summed data track and not the normalised data
        #track, to overweight those positions with more observations
        array = grammar.get_summed_data_track(self.trackNameToUse);
        #recall: grammarArray.shape = (4, 61)
        #find the sum at each position
        if (self.useRangeNotSum):
            valPerPosition = np.max(array, axis=0) - np.min(array, axis=0) 
        else:
            valPerPosition = np.sum(array, axis=0); 
        slidingWindowSums = util.computeRunningWindowSum(valPerPosition
                                , self.slidingWindowSizeForPeak); 
        maxPos = np.argmax(slidingWindowSums);
        startPos = max(0,maxPos-self.flanksToExpand)
        endPos = min(array.shape[1], maxPos+self.slidingWindowSizeForPeak+self.flanksToExpand);
        return (startPos, endPos); 

def printLabelAndModiscoMotif(grammars, **kwargs):
    if isinstance(grammars, list):
        grammars = dict(enumerate(grammars));
    for (label, grammar) in grammars.items():
        printModiscoMotif(grammar
                     , title="grammar "+str(label)
                            +", totalObservationsEver:"+str(grammar.totalObservationsEver)
                     , **kwargs);

def printModiscoMotifWithIdx(grammars, idx, *args, **kwargs):
    """
        convencience function that calls printModiscoMotif on the
            specified index and generates the title accordingly
    """ 
    printModiscoMotif(grammars[idx], title=idx, *args, **kwargs);
    #printModiscoMotif(grammars[idx].getRevCompModiscoMotif(), title=idx, *args, **kwargs);

def printModiscoMotif(grammar, trackNamesToPrint, heightPerTrack=3, minObs=0, plotPosEvery=1, title="default title"): 
    import matplotlib.pyplot as plt;
    import matplotlib.gridspec as grd;
    
    num_underlying_observations=grammar.num_underlying_observations.astype("int")
    #find first index with minObs observations:
    idxsPassingNumObsThreshold = [i for (i,val) in enumerate(num_underlying_observations)
                                                                        if val>=minObs];
    assert len(idxsPassingNumObsThreshold) >= 1,\
        "only "+str(len(idxsPassingNumObsThreshold))\
        +" positions have at least "+str(minObs)+" underlying observations";
    firstIdx = idxsPassingNumObsThreshold[0];
    lastIdx = idxsPassingNumObsThreshold[-1]+1
    num_underlying_observations=num_underlying_observations[firstIdx:lastIdx];

    plt.clf()
    width = lastIdx-firstIdx
    fig_width = 20 + width/10
    fig = plt.figure(figsize=(fig_width,heightPerTrack*len(trackNamesToPrint)))
    for (i,trackName) in enumerate(trackNamesToPrint):
        ax = fig.add_subplot(len(trackNamesToPrint),1,i+1)
        arr = grammar.getNormalisedDataTrack(trackName);
        summedDataTrack = grammar.summedDataTracks[trackName]
        arr = arr[:,firstIdx*summedDataTrack.effectiveStride
                    :(lastIdx-1)*summedDataTrack.effectiveStride+summedDataTrack.effectiveWidth];
        if (arr.shape[0]==4):
            letter_heights=arr.T
            pos_heights = np.copy(letter_heights)
            pos_heights[letter_heights < 0] = 0
            neg_heights = np.copy(letter_heights)
            neg_heights[letter_heights > 0] = 0
            for x_pos, heights in enumerate(letter_heights):
                letters_and_heights = sorted(deepLIFTutils.izip(heights, 'ACGT'))
                y_pos_pos = 0.0
                y_neg_pos = 0.0
                for height, letter in letters_and_heights:
                    if height > 0:
                        deepLIFTutils.add_letter_to_axis(ax, letter, -0.5+x_pos, y_pos_pos, height)
                        y_pos_pos += height
                    else:
                        deepLIFTutils.add_letter_to_axis(ax, letter, -0.5+x_pos, y_neg_pos, height)
                        y_neg_pos += height
            if (i==len(trackNamesToPrint)):
                ax.set_xlabel('pos')
            ax.set_aspect(aspect='auto', adjustable='box')
            ax.autoscale_view()
        elif (arr.shape[0]==1):        
            ax.plot(range(arr.shape[1]), arr.squeeze(), 'k', lw=0.5)
            ax.axhline(0, linestyle='dashed', color='black')
        else:
            mplh.plotHeatmapGivenAx(ax, data=arr , logTransform=False
                                    , zeroCenter=True
                                    , cmap=plt.cm.coolwarm);
        #else:
        #    raise RuntimeError("Unsure how to deal with shape "+str(arr.shape));
        ax.set_ylabel(trackName)
        ax.set_xlim(-1, arr.shape[1]+1)
        xticks_locations=range(-1,arr.shape[1]+1)
        ax.set_xticks(xticks_locations[::plotPosEvery])
        ax_2=ax.twiny()
        numObsSpacing=plotPosEvery+1
        ax_2.set_xticks(xticks_locations[::numObsSpacing])
        ax_2.set_xticklabels(num_underlying_observations[::numObsSpacing])
        if (i==0):
            ax_2.set_xlabel('numObs')
    #plt.title(title)
    plt.show();
             
    """
    if (grammar.normedCoreDeepLIFTtrack.shape[0]==4):
        deepLIFTutils.plotWeights(
            weights=grammar.normedCoreDeepLIFTtrack.T, bias=0
            ,title=str(title));
    elif (grammar.normedCoreDeepLIFTtrack.shape[0]==5):
        deepLIFTutils.plot_sequenceAndSignal(
            grammar.normedCoreDeepLIFTtrack
            ,title=str(title)
            ,additionalX=num_underlying_observations
            ,additionalXSpacing=int(np.ceil(len(num_underlying_observations)/30))) 
    else:
        raise RuntimeError("Not sure how to handle",grammar.normedCoreDeepLIFTtrack);
    """

def saveModiscoMotifsToPkl(grammars, pklFile):
    import pickle
    reload(pickle)
    toPkl = [(x.summedDataTracks
              , x.num_underlying_observations
              , x.totalObservationsEver)
              for x in grammars];
    pickle.dump(toPkl, open(pklFile,'w'));

def loadModiscoMotifsFromPkl(pklFile):
    """
        expecting pickled file that contains a list of
            (numpyArray, num_underlying_observations)...
    """
    grammarsRawData = pickle.load(pklFile);
    return [ModiscoMotif(summedDataTracks=x[0]
                    , num_underlying_observations=x[1]
                    , totalObservationsEver=x[2])
                for x in grammarsRawData]; 

def getTopNGreedyNonOverlappingCorrScores(
    largerArr, smallerArr, rev_comp_func
    , N, excludeHitsWithinWindow
    , normaliseFunc
    , smallerPerPosNormFuncs
    , largerPerPosNormFuncs
    , auxLargerForPerPosNorm
    , auxLargerPerPosNormFuncs
    , smallerIsPalindrome):
    """
        will greedily take the best positions
            and will ignore hits within exclude_peaks_within_window
            of an already-included position
        auxLargerForPerPosNorm is intended for sequence.
            idea is to take [gradientMatch/maxMatch (computable from gradient)]*deepLIFTstrength
            in other words: gradient match is "how well does this motif match what was being looked for"
                            and deepLIFTstrength/maxMatch is "how much of what was being looked for was gotten"
        smallerIsPalindrome: if True, will not distinguish between forward and reverse orientation hits (will
            take the max of both and will declare everything to be in the forward orientation)
        Returns: scores, positions, fwd, leftIdxs, rightIdxs - each is an array of length N 
        fwd = 1 if in fwd orientation, -1 if in reverse
        leftIdxs and rightIdxs are the left and right indexes of the hit
        positions = leftIdx + (length of smallerArr) if fwd = -1, else leftIdx; think of it as
            corresponding to where the front of the protein would contact the sequence (where the
            front is the defined as the part of the protein that contacts the left end of the motif
            hit when it is in the fwd orientation). The adjustment is useful for grammar detection.
    """
    #...some inefficiency if the normalisePerPos is being done twice.
    #actually possibly more than a little inefficiency.
    crossCorrelations_fwd, firstIsSmaller, smallerLen =\
    util.crossCorrelateArraysLengthwise(largerArr, smallerArr
                                        , normaliseFunc=normaliseFunc
                                        , smallerPerPosNormFuncs=smallerPerPosNormFuncs
                                        , largerPerPosNormFuncs=largerPerPosNormFuncs
                                        , auxLargerForPerPosNorm=auxLargerForPerPosNorm
                                        , auxLargerPerPosNormFuncs=auxLargerPerPosNormFuncs);
    crossCorrelations_rev, firstIsSmaller, smallerLen =\
        util.crossCorrelateArraysLengthwise(largerArr, rev_comp_func(smallerArr)
                                             , normaliseFunc=normaliseFunc
                                             , smallerPerPosNormFuncs=smallerPerPosNormFuncs
                                             , largerPerPosNormFuncs=largerPerPosNormFuncs
                                             , auxLargerForPerPosNorm=auxLargerForPerPosNorm
                                             , auxLargerPerPosNormFuncs=auxLargerPerPosNormFuncs);
    assert firstIsSmaller==False;
    crossCorrelations=np.maximum(crossCorrelations_fwd, crossCorrelations_rev);
    if (not smallerIsPalindrome):
        hitIsFwd = 1*(crossCorrelations_fwd >= crossCorrelations_rev) + -1*(crossCorrelations_fwd < crossCorrelations_rev);
    else:
        hitIsFwd = np.ones(crossCorrelations.shape);
    
    scores, positions, fwd, leftIdxs, rightIdxs = [], [], [], [], [];
    if (N==1):
        bestIdx = np.argmax(crossCorrelations);
        scores.append(crossCorrelations[bestIdx]);
        isFwd = hitIsFwd[bestIdx];
        #This shift is necessary because cross correlation pads in front by smallerLen-1
        actualPos = bestIdx - (smallerLen-1) 
        leftIdxs.append(actualPos)
        rightIdxs.append(actualPos+smallerLen)
        positions.append(actualPos+(0 if isFwd==1 else smallerLen));
        fwd.append(isFwd);
    else:
        #sort scores 
        sortedScores = sorted(enumerate(crossCorrelations), key=lambda x: -x[1]); 
        i = 0;
        idxs = []
        while len(scores) < N and i < len(sortedScores):
            #only consider the idx if it is not within
            #exclude_peaks_within_window of any of the included
            #idxs
            skip = any([abs(sortedScores[i][0]-x) <= excludeHitsWithinWindow
                        for x in idxs]);
            if (not skip):
                scores.append(sortedScores[i][1]);
                isFwd = hitIsFwd[sortedScores[i][0]]
                idxs.append(sortedScores[i][0]);
                fwd.append(isFwd);
            i += 1; 
        #the shift of (smallerLen-1) is necessary because cross correlation pads by smallerLen-1
        #in front
        positions = [x + (0 if hitIsFwd[x]==1 else smallerLen) for x in idxs]; 
        positions = [x-(smallerLen-1) for x in positions];
        leftIdxs = [x-(smallerLen-1) if x is not None else None for x in idxs]; 
        rightIdxs = [x + smallerLen for x in leftIdxs]
        if (len(scores) < N):
            print("Warning: you wanted",N,"scores, but with "
                "your excludeHitsWithinWindow setting of"
                ,excludeHitsWithinWindow,"and largerArr len"
                ,largerArr.shape[1],"could not get more than "
                ,len(scores),"peaks at positions",positions,
                " - in the meantime I will fill the rest with 0");
            leftIdxs.extend([None]*(N-len(scores)))
            rightIdxs.extend([None]*(N-len(scores)))
            positions.extend([None]*(N-len(scores)));
            scores.extend([0]*(N-len(scores)));
            fwd.extend([True]*(N-len(scores)));
    assert firstIsSmaller==False;    
    #if you change the line below, please also change
    #recastOutputOfTopNGreedy accordingly!
    return scores, positions, fwd, leftIdxs, rightIdxs;
 
###################################################
#I have to endure this nonsense because the function
#pickled by Pool.map has to be accessible at the top level
_topNgreedy_largerArrs = util.VariableWrapper(None);
_topNgreedy_smallerArrs = util.VariableWrapper(None);
_topNgreedy_rev_comp_func = util.VariableWrapper(None);
_topNgreedy_N = util.VariableWrapper(None);
_topNgreedy_excludeHitsWithinWindow = util.VariableWrapper(None);
_topNgreedy_normaliseFunc = util.VariableWrapper(None);
_topNgreedy_smallerPerPosNormFuncs=util.VariableWrapper(None);
_topNgreedy_largerPerPosNormFuncs=util.VariableWrapper(None);
_topNgreedy_auxLargerForPerPosNorm = util.VariableWrapper(None);
_topNgreedy_auxLargerPerPosNormFuncs = util.VariableWrapper(None);
_topNgreedy_palindromes = util.VariableWrapper(None);
#Nonsense endured
####################################################
def getTopNGreedyNonOverlappingCorrScores_forParallel(i):
    smallerArrCorrs = [
        getTopNGreedyNonOverlappingCorrScores(
            largerArr=_topNgreedy_largerArrs.var[i]
            , smallerArr=smallerArr
            , rev_comp_func=_topNgreedy_rev_comp_func.var
            , N=_topNgreedy_N.var
            , excludeHitsWithinWindow=
                _topNgreedy_excludeHitsWithinWindow.var
            , normaliseFunc=_topNgreedy_normaliseFunc.var
            , smallerPerPosNormFuncs=_topNgreedy_smallerPerPosNormFuncs.var
            , largerPerPosNormFuncs=_topNgreedy_largerPerPosNormFuncs.var
            , auxLargerForPerPosNorm=None if _topNgreedy_auxLargerForPerPosNorm.var is None
                                        else _topNgreedy_auxLargerForPerPosNorm.var[i]
            , auxLargerPerPosNormFuncs=_topNgreedy_auxLargerPerPosNormFuncs.var
            , smallerIsPalindrome=(smallerArrIdx in _topNgreedy_palindromes.var)
            )
        for (smallerArrIdx, smallerArr) in enumerate(_topNgreedy_smallerArrs.var)
    ]; 
    return smallerArrCorrs;

Hit = namedtuple("Hit", ["score", "pos", "fwd", "leftIdx", "rightIdx","motifIdx","inputIdx","grammarIdx"])
#the defaults for leftIdx/rightIdx/motifIdx/grammarIdx/inputIdx are 'None'
Hit.__new__.__defaults__ = (None,None,None,None,None); 
#"grammarIdx" is only there for back-compat with a time where "motifs" were called "grammars"
#see documentation of getTopNGreedyNonOverlappingCorrScores for pos vs fwd vs leftIdx vs rightIdx
def recastOutputOfTopNGreedy(outputOfTopNgreedy):
    """
        Recasts the output of getTopNGreedyNonOverlappingCorrScores_onFullSet
            input is: [num examples x num motifs x 5 x N]
        to return something like this:
            [num of motifs x num examples x N (as in the "topN" scores)]
        Each entry of the third dimension is a "Hit" object (see above)
    """
    hitsForDifferentMotifs = [[] for i in range(outputOfTopNgreedy.shape[1])]
    for (inputIdx,example) in enumerate(outputOfTopNgreedy):
        for (motifNumber, motifHits) in enumerate(example):
            hitsForThisMotif = [] #will store the motif hits in a nice format
            for hitNumber in range(len(motifHits[0])):
                hitsForThisMotif.append(Hit(score=motifHits[0][hitNumber]
                                    ,pos=motifHits[1][hitNumber]
                                    ,fwd=motifHits[2][hitNumber]
                                    ,leftIdx=motifHits[3][hitNumber]
                                    ,rightIdx=motifHits[4][hitNumber]
                                    ,motifIdx=motifNumber
                                    ,inputIdx=inputIdx)); 
            hitsForDifferentMotifs[motifNumber].append(hitsForThisMotif)
    return hitsForDifferentMotifs;

def getTopNGreedyNonOverlappingCorrScores_onFullSet(
        largerArrs, smallerArrs, rev_comp_func
        , N, excludeHitsWithinWindow
        , normaliseFunc=util.CROSSC_NORMFUNC.none
        , smallerPerPosNormFuncs=[]
        , largerPerPosNormFuncs=[]
        , auxLargerForPerPosNorm=None
        , auxLargerPerPosNormFuncs=[]
        , palindromes={}
        , secondsBetweenUpdates=1, numThreads=1):
    """
        largerArrs: regions to get the corr scores on
        smallerArrs: regions to correlate with largerArrs
        rev_comp_func: function for reverse complementation
        N, excludeHitsWithinWindow: see docs for
            getTopNGreedyNonOverlappingCorrScores 
        Returns something of the following dimensions:
            [num examples x number of motifs x 5 x N (as in the "topN" scores; the first index is the highest score)]
            Regarding the third dimension which is of length 3, the indexes are as follows:
            Index 0 = the actual score
            Index 1 = left index of the hit if in fwd orientation, left index + motifLen if hit was in reverse orientation. (This adjustment is useful for grammar detection)
            Index 2 = 1 if hit was in forward orientation and -1 if hit was in reverse orientation
            Index 3 = left index of hit
            Index 4 = right index of hit
            The runtime scales linearly with N so I suggest setting N=1 if you can.
    """
    assert auxLargerForPerPosNorm is None or auxLargerForPerPosNorm.shape==largerArrs.shape
    startTime=time.time();
    _topNgreedy_largerArrs.var=largerArrs
    _topNgreedy_smallerArrs.var=smallerArrs
    _topNgreedy_rev_comp_func.var=rev_comp_func
    _topNgreedy_N.var = N
    _topNgreedy_excludeHitsWithinWindow.var = excludeHitsWithinWindow
    _topNgreedy_normaliseFunc.var = normaliseFunc
    _topNgreedy_smallerPerPosNormFuncs.var = smallerPerPosNormFuncs
    _topNgreedy_largerPerPosNormFuncs.var = largerPerPosNormFuncs
    _topNgreedy_auxLargerForPerPosNorm.var = auxLargerForPerPosNorm
    _topNgreedy_auxLargerPerPosNormFuncs.var = auxLargerPerPosNormFuncs
    _topNgreedy_palindromes.var = palindromes
    if (numThreads > 1):
        toReturn = util.multiprocessing_map_printProgress(
                    secondsBetweenUpdates=secondsBetweenUpdates
                    ,numThreads=numThreads
                    ,func=getTopNGreedyNonOverlappingCorrScores_forParallel
                    ,iterable=range(len(largerArrs)));
    else:
        toReturn = [];
        for i in range(len(largerArrs)):
            toReturn.append(getTopNGreedyNonOverlappingCorrScores_forParallel(i));
            if (i%1000==0):
                print("Done",i); 
    print("Time taken:",time.time()-startTime);
    return toReturn;

def extractScoresOnlyFromHitsMatrix(hitsMatrix, topNtoKeep):
    """
        hitsMatrix has dimensions:
            numExamples x numMotifs x 2 x N
    """
    return np.array(hitsMatrix)[:,:,0,:topNtoKeep]
   
class ReshapeCorrScoresInto2Dmatrix(object):
    def __init__(self, topNtoKeep):
        self.topNtoKeep = topNtoKeep;
    def __call__(self, hitsMatrix):
        """
            hitsMatrix has dimensions:
                numExamples x numMotifs x 2 x N
            first index in third dimension corresponds to scores, the
                second index corresponds to the positions of the scores.
            Extract the first index and reshape into numExamples x (numMotifs*N)
        """
        raise NotImplementedError();
 
class ReshapeCorrScoresInto2Dmatrix_normalisePerMotif(ReshapeCorrScoresInto2Dmatrix):
    def __call__(self, hitsMatrix):
        """
            Normalise each motif's scores by the mean and standard deviation over all
                hits to that motif (even accross multiple ranks)
        """
        scoresOnly = extractScoresOnlyFromHitsMatrix(hitsMatrix, self.topNtoKeep);
        matrixToNormaliseByColumns =   np.transpose(scoresOnly, axes = (1, 0, 2))\
                                                .reshape((len(hitsMatrix[0]),-1)) 
        stdevPerMotif =  np.std(matrixToNormaliseByColumns, axis=1);
        meanPerMotif = np.mean(matrixToNormaliseByColumns, axis=1);
        assert stdevPerMotif.shape == (len(hitsMatrix[0]),);
        #normalise the scores by mean and sdev
        scoresOnly = (scoresOnly-meanPerMotif[None,:,None]) / stdevPerMotif[None,:,None]
        #scoresOnly = (scoresOnly) / stdevPerMotif[None,:,None]
        #reshape into 2D matrix
        return np.reshape(scoresOnly, (scoresOnly.shape[0]
                                        , scoresOnly.shape[1]*scoresOnly.shape[2]));

class ReshapeCorrScoresInto2Dmatrix_normaliseBySdevPerColumn(ReshapeCorrScoresInto2Dmatrix):
    def __call__(hitsMatrix):
        """
            Normalise by the sdev of each column, after the reshape
                to the 2D matrix.
        """
        scoresOnly = extractScoresOnlyFromHitsMatrix(hitsMatrix, self.topNtoKeep);
        #reshape into 2D matrix
        scoresOnly_reshaped = np.reshape(scoresOnly, (scoresOnly.shape[0]
                                        , scoresOnly.shape[1]*scoresOnly.shape[2]));
        #normalise columns by mean + stdev
        return (scoresOnly_reshaped - np.mean(scoresOnly_reshaped, axis=1))\
                                    /np.std(scoresOnly_reshaped, axis=1)

def obtain2DscoresForAllLabelsSatisfying(motifHitsSets, datas
                                        ,labelCriterion  
                                        ,twoDscoreGetterFunc):
    twoDscores = (twoDscoreGetterFunc(motifHits) for motifHits in motifHitsSets)
    #subset the hits according to labelCriterion and concat
    twoDscores = np.array(list(itertools.chain(*[itertools.compress(hits
                        , (labelCriterion(y) for y in data.Y))
                        for hits, data in zip(twoDscores, datas)])))
    ids = list(itertools.chain(*[itertools.compress(data.ids
                                , (labelCriterion(y) for y in data.Y))
                                for data in datas]))
    return twoDscores, ids;
    
def get_seqletsConsideringFilterSubset(filterArrayOfContribs
                                        , rawSequenceArrayOfContribs
                                        , indexesOfFiltersToConsider
                                        , indicesToGetSeqletsOn
                                        , segment_identifier
                                        #kernelWidthsAndStrideWidths:
                                        #first indices correspond to earlier layers.
                                        #If no earlier conv layers, is a list with 1 tuple
                                        , kernelAndStrideWidths
                                        , includeNeg
                                        , numThreads
                                        , secondsBetweenUpdates
                                        , rev_comp_func=None):
    """
        Is for identifying seqlets associated with specific filters
    """
    if (rev_comp_func is None):
        rev_comp_func=RevCompWithDNArowsSubset(dnaRowsStart=0, dnaRowsEnd=4);
        print("No reverse comp function provided so assuming you have dna as first 4 rows");
        
    #"filterArrayOf..." is the same as for get_seqlets. Has shape
    # example x channel x rows x len
    
    assert len(filterArrayOfContribs.shape)==4
    assert filterArrayOfContribs.shape[2]==1
    assert len(rawSequenceArrayOfContribs.shape)==4;
    assert rawSequenceArrayOfContribs.shape[1]==1;
    assert rawSequenceArrayOfContribs.shape[2]==4;
    #reshape to drop out the channel axis from the raw seq contribs
    reshapedRawSequenceContribs = np.squeeze(rawSequenceArrayOfContribs)
    if (includeNeg==False):
        #apply a mask for only positive contribs
        reshapedRawSequenceContribs = reshapedRawSequenceContribs*\
                                        (reshapedRawSequenceContribs>0);
    
    #compute the effective filter width/stride in terms of raw sequence
    kernelAndStrideWidths = kernelAndStrideWidths[::-1]
    effectiveFilterWidth, effectiveFilterStride = kernelAndStrideWidths[0]
    for (kernWidPrevLyr, strideWidPrevLyr) in kernelAndStrideWidths[1:]:
        effectiveFilterWidth = kernWidPrevLyr + (effectiveFilterWidth-1)*strideWidPrevLyr
        effectiveFilterStride *= strideWidPrevLyr

    #subset to contributions from specific filters of interest
    filterArrayOfContribs = filterArrayOfContribs[:,indexesOfFiltersToConsider]
    
    #find sections of importance
    #filterKeySegments will hold the start and end index in terms of the filter layer's length axis
    print("filterArrayOfContribs.shape",filterArrayOfContribs.shape)
    filterSeqlets, filterSeqletIndices = get_seqlets(
                        raw_importance_contribs=filterArrayOfContribs
                        ,indicesToGetSeqletsOn=indicesToGetSeqletsOn
                        ,outputsBeforeActivation=None #not needed if threshold is 1
                        ,activation=None #not needed if threshold is 1
                        ,thresholdProb=1.0
                        ,segment_identifier=segment_identifier                            
                        ,rev_comp_func=reverseFunc #does not really matter for filters
                        ,includeNeg=includeNeg
                        ,numThreads=numThreads
                        ,secondsBetweenUpdates=secondsBetweenUpdates)
    sequenceSeqlets = [];
    for filterSeqlet in filterSeqlets:
        (filterLocStart, filterLocEnd) = filterSeqlet.location;
        (seqLocStart, seqLocEnd) = filterLocStart*effectiveFilterStride\
                                    , filterLocEnd*effectiveFilterStride\
                                      + effectiveFilterWidth;
        assert seqLocStart < seqLocEnd;
        assert seqLocEnd <= reshapedRawSequenceContribs.shape[2];
        summedDataTracks={ModiscoMotif.core_importance_track:
                            DataTrack(data=reshapedRawSequenceContribs\
                                           [filterSeqlet.sequenceId,:,seqLocStart:seqLocEnd]
                                      ,pseudocount=0
                                      ,rev_comp_func=rev_comp_func)};
        sequenceSeqlet = Seqlet(
                            summedDataTracks=summedDataTracks
                            ,num_underlying_observations=1
                            ,totalObservationsEver=1
                            ,location=(seqLocStart,seqLocEnd)
                            ,sequenceId=filterSeqlet.sequenceId)    
        sequenceSeqlets.append(sequenceSeqlet);
    return sequenceSeqlets, filterSeqletIndices;

def getTopFiltersByImportance(filterScores_forClustering, indicesSubset, topNFilters):
    summedFilterImportancesForIndices =\
               np.sum(np.array([filterScores_forClustering[x]
                         for x in indicesSubset]), axis=0);
    rankedFilterImportances = sorted(enumerate(summedFilterImportancesForIndices)
                                     , key=lambda x: -x[1]);
    indexesOfFiltersToConsider = [x[0] for x in rankedFilterImportances[:topNFilters]];
    return indexesOfFiltersToConsider;

def get_seqletsForSpecificFilterSubsets(filtClustLabelToIndicesWithinClusteredArr
                                        , correspondingIndicesIntoValidArr
                                        , dLValidRawFilterContribs_singleNeuron
                                        , dLValidRawSequenceContribs_singleNeuron
                                        , filterScores_forClustering
                                        , segment_identifier
                                        , kernelAndStrideWidthsOfPrevLayers
                                        , rev_comp_func
                                        , topNFilters=None
                                        , specificFilters=None): 
    """
        filterScores_forClustering: 2d matrix of deepLIFT sores on true positives; for each
            region the scores for a particular filter are summed lengthwise
        filtClustLabelToIndicesWithinClusteredArr: dict from filter cluster label to indices within
            filterScores_forClustering
        returns: filtClustLabelToSeqletsAndIndices, which is a dict of
                    label -> (seqletsForFilterCluster, seqletIndices)
                    here seqlet indices refers to index within original valid set
    """
    #topNFilters and specificFilters are mutually exclusive options
    assert topNFilters is None or specificFilters is None;
    assert topNFilters is not None or specificFilters is not None;

    filtClustLabelToSeqletsAndIndices = OrderedDict(); 
    for filtClustLabel in sorted(filtClustLabelToIndicesWithinClusteredArr.keys()):
        indicesWithinClusteredArr = filtClustLabelToIndicesWithinClusteredArr[filtClustLabel];
        #map the index with respect to truePositiveIndices to the index into the
        #full validation set itself.
        indicesWithinValidArrForCluster = [correspondingIndicesIntoValidArr[x]\
                                            for x in indicesWithinClusteredArr];
        topFilterIndices = specificFilters if specificFilters is not None\
                                else getTopFiltersByImportance(
                                        filterScores_forClustering=
                                            filterScores_forClustering
                                        , indicesSubset=indicesWithinClusteredArr
                                        , topNFilters=topNFilters)
        print("Running on filters",topFilterIndices);
        
        #seqletIndices are into the original validation dataset array
        seqletsForFilterCluster, seqletIndices = get_seqletsConsideringFilterSubset(
            filterArrayOfContribs=dLValidRawFilterContribs_singleNeuron
            , rawSequenceArrayOfContribs=dLValidRawSequenceContribs_singleNeuron
            , indexesOfFiltersToConsider=topFilterIndices
            , indicesToGetSeqletsOn=indicesWithinValidArrForCluster
            , segment_identifier=segment_identifier
            , kernelAndStrideWidths=kernelAndStrideWidthsOfPrevLayers
            , rev_comp_func=rev_comp_func
            , numThreads=2
            , secondsBetweenUpdates=1
        )
        filtClustLabelToSeqletsAndIndices[filtClustLabel] = (seqletsForFilterCluster, seqletIndices);
    return filtClustLabelToSeqletsAndIndices;

PairDistance = namedtuple("PairDistance", ["rowIdx","hit1","hit2","sep"])
def obtainPairwiseDistancesBetweenHits(hitsForRows):
    pairwiseDistances = defaultdict(lambda: defaultdict(list))
    for (rowIdx, hitsForRow) in enumerate(hitsForRows):
        for (hit1Idx, hit1) in enumerate(hitsForRow):
            for (hit2Idx, hit2) in enumerate(hitsForRow[hit1Idx+1:]):
                #the smaller hit will always come first due to the ordering of hitsForRow
                if (hit1.grammarIdx == hit2.grammarIdx):
                    if (hit1.fwd==-1 and hit2.fwd==1):
                        hit1,hit2 = hit2,hit1 #swap
                pairDistanceObject = PairDistance(
                                        rowIdx=rowIdx
                                        ,hit1=hit1
                                        ,hit2=hit2
                                        ,sep=hit2.pos-hit1.pos)
                pairwiseDistances[str(hit1.grammarIdx)+"_"+str(hit1.fwd)][str(hit2.grammarIdx)+"_"+str(hit2.fwd)].append(pairDistanceObject);
    return pairwiseDistances;

def obtainHitsForRows(topHitsForEachRow, zScoreThresholdForHit):
    topHitsForEachRow=topHitsForEachRow.copy()
    #topHitsForEachRow is:
    #numExamples x grammar x score,idx,fwd x top N hits
    assert len(topHitsForEachRow.shape)==4;
    meanPerModiscoMotif = np.mean(topHitsForEachRow[:,:,0,:],axis=(0,-1))
    stdPerModiscoMotif = np.std(topHitsForEachRow[:,:,0,:],axis=(0,-1))
    topHitsForEachRow[:,:,0,:] =\
        (topHitsForEachRow[:,:,0,:]-meanPerModiscoMotif[None,:,None])/stdPerModiscoMotif[None,:,None]
    for i in range(topHitsForEachRow.shape[1]):
        mplh.plotHist(np.ravel((topHitsForEachRow[:,0,0,:])), bins=50)
    hitsForRows = [];
    for (topHitsRow) in topHitsForEachRow:
        hitsForRow = []
        for (grammarIdx, grammarHits) in enumerate(topHitsRow):
            for (grammarHitIdx,grammarHitScore) in enumerate(grammarHits[0]):
                if (grammarHitScore > zScoreThresholdForHit[grammarIdx]):
                    hitsForRow.append(Hit(grammarIdx=grammarIdx
                                          , score=grammarHitScore
                                          , pos=grammarHits[1][grammarHitIdx]
                                          , fwd=grammarHits[2][grammarHitIdx]))
        hitsForRows.append(hitsForRow) 
    return hitsForRows;

def plotPairwiseDistances(pairwiseDistances, oneHotSequenceData, *args, **kwargs):
    for grammar1idx, grammar1key in enumerate(sorted(pairwiseDistances.keys())):
        for grammar2key in sorted(pairwiseDistances[grammar1key].keys()):
            plotPairwiseDistance_specificPair(pairwiseDistances, oneHotSequenceData, grammar1key, grammar2key, *args, **kwargs);

def plotPairwiseDistance_specificPair(pairwiseDistances, oneHotSequenceData, grammar1key, grammar2key, flanksToPlot=60, topNtoListOut=10
                                        , topNtoPlotSeqFor=2, numSeqToPlot=5):
    oneHotSequenceData=np.squeeze(oneHotSequenceData);
    assert len(oneHotSequenceData.shape)==3;
    assert oneHotSequenceData.shape[1]==4;
    pairDistanceObjects = sorted(pairwiseDistances[grammar1key][grammar2key],key=lambda x: x.sep);
    dataForHistogram=[x.sep for x in pairDistanceObjects]
    if (len(dataForHistogram)>0):
        thePair=str(grammar1key)+" and "+str(grammar2key)
        print("Key positions for "+thePair)
        counterItems = sorted(Counter(dataForHistogram).items(), key=lambda x: -x[1])[:topNtoListOut] 
        print(counterItems)
        mplh.plotHist(dataForHistogram
                      ,title=thePair
                      ,bins=200
                      ,figsize=(14,7))
        #for each 
        sequenceDataToPlot = [];
        for pairDistanceObject in pairDistanceObjects:
            if (pairDistanceObject.sep <= flanksToPlot):
                seqToPutIn = np.zeros((4,2*flanksToPlot));
                rowIdx = pairDistanceObject.rowIdx;   
                hit1start = pairDistanceObject.hit1.pos
                #before/after may not span the entire seq hence all this
                before = oneHotSequenceData[rowIdx,:,(hit1start-flanksToPlot):hit1start];
                after = oneHotSequenceData[rowIdx,:,hit1start:(hit1start+flanksToPlot)]; 
                seqToPutIn[:,flanksToPlot-before.shape[1]:flanksToPlot] = before;
                seqToPutIn[:,flanksToPlot:flanksToPlot+after.shape[1]] = after;
                sequenceDataToPlot.append(seqToPutIn);
        mplh.plotOneHotEncodingsAsImage(np.array(sequenceDataToPlot));
        for (separation, count) in counterItems[:topNtoPlotSeqFor]:
            plotSequencesForSeparations(pairwiseDistances, oneHotSequenceData, grammar1key, grammar2key, [separation], numSeqToPlot, flanksToPlot);

def plotSequencesForSeparations(pairwiseDistances, oneHotSequenceData, grammar1key, grammar2key, separations, numSeqToPlot, flanksToPlot):
    pairDistanceObjects = pairwiseDistances[grammar1key][grammar2key];
    for separation in separations:
        print("plotting separation",separation);
        #filter for all pair distance objects with that separation
        pairDistanceObjectsWithSep = [x for x in pairDistanceObjects if x.sep==separation];
        #sort them in order of strongest sum of hits first
        pairDistanceObjectsWithSep = sorted(pairDistanceObjectsWithSep, key=lambda x: -(min(x.hit1.score,x.hit2.score)));
        #plot the seq!
        for pairDistanceObject in pairDistanceObjectsWithSep[:numSeqToPlot]:
            hit1pos = pairDistanceObject.hit1.pos;
            hit2pos = pairDistanceObject.hit2.pos;
            assert hit2pos-hit1pos==separation;
            start = min(hit1pos, hit2pos)-flanksToPlot;
            end = max(hit1pos, hit2pos)+flanksToPlot; 
            seqToPlot = oneHotSequenceData[pairDistanceObject.rowIdx
                                            , :, start:end];
            deepLIFTutils.plot_bases(seqToPlot.T,figsize=(20,1)); 

def compareToKnownMotifs(mergedModiscoMotifs, trackNameForComparison=ModiscoMotif.core_importance_track):
    import compare_filters_to_known_motifs
    reload(compare_filters_to_known_motifs)
    pwms=compare_filters_to_known_motifs.load_all_pwms()
    for (i,grammar) in mergedModiscoMotifs.items():
        print("We are on grammar",i,"\n");
        for pwmSet in pwms:
            hits=compare_filters_to_known_motifs.get_pwm_matches_for_filter(grammar.getNormalisedDataTrack(trackNameForComparison),pwmSet)
            print(hits)
            print("")  

ConvLayerTypes = util.enum(conv='conv', maxpool='maxpool')
ConvLayerDetails = namedtuple("ConvLayerDetails", ["layerType", "stride", "width", "weights", "biases", "activation"])
def kerasLayerToConvLayerDetails(kerasLayer, activation):
    config = kerasLayer.get_config()
    if config['custom_name']=='convolution2d':
        return ConvLayerDetails(
                layerType=ConvLayerTypes.conv
                , stride=config['subsample'][1]
                , width=config['nb_col']
                , weights=kerasLayer.get_weights()[0]
                , biases=kerasLayer.get_weights()[1]
                , activation=activation)
    else:
        raise RuntimeError("Unsupported custom_name: "+str(config['custom_name']));

def determineFinalLayerEffectiveStrideAndWidth(convLayersDetails):
    finalEffectiveStride = convLayersDetails[0].stride
    finalEffectiveWidth = convLayersDetails[0].width
    for convLayerDetails in convLayersDetails[1:]:
        finalEffectiveWidth = finalEffectiveWidth + (convLayerDetails.width-1)*finalEffectiveStride; 
        finalEffectiveStride *= convLayerDetails.stride;
    return finalEffectiveStride, finalEffectiveWidth;

def compileMiniKerasModel(convLayersDetails, inputSize, finalOutputWeights=None):
    assert len(convLayersDetails)==1; #only handling 1 for now
    assert all([x.stride==1 for x in convLayersDetails]); #only stride 1 for now
    assert convLayersDetails[0].weights.shape[1]==1; #input has 1 channel
    assert convLayersDetails[0].weights.shape[2]==4; #input has 4 rows
    
    from keras.models import Sequential
    from keras.layers.core import Flatten
    from keras.layers.core import Dense
    
    model = Sequential();
    for (i,convLayerDetails) in enumerate(convLayersDetails):
        assert int(inputSize)==inputSize;
        input_shape=(1,4,int(inputSize)) if i==0 else None;
        convLayer = createKerasConvLayerFromConvLayerDetails(
                        convLayerDetails, input_shape=input_shape); 
        model.add(convLayer);
    if (finalOutputWeights is not None):
        model.add(Flatten()); 
        denseLayer = Dense(1,activation="linear");
        model.add(denseLayer);
    model.compile(loss='mse', optimizer='sgd');
    for (i,convLayerDetails) in enumerate(convLayersDetails): 
        model.layers[i].set_weights([convLayerDetails.weights, convLayerDetails.biases])
    model.layers[-1].set_weights([finalOutputWeights.ravel()[:,None],np.array([0])]);
    return model;

def createKerasConvLayerFromConvLayerDetails(convLayerDetails, input_shape=None): 
    from keras.layers.convolutional import Convolution2D;
    convLayer = Convolution2D(nb_filter=convLayerDetails.weights.shape[0]
                #weights shape = num channels x num channels of previous layer
                #                x kernel width x kernel height
                            ,nb_row=convLayerDetails.weights.shape[2] 
                            ,nb_col=convLayerDetails.width
                            ,activation=convLayerDetails.activation
                            ,**{} if input_shape is None else {'input_shape':input_shape})
    return convLayer;


#puts in a bunch of defaults
def get_correlation_matrix_diffRowsAndCols_shortcut(
    rowMotifs, colMotifs, subtracks_to_align_onForCC, batch_size,
    verbose=True):
    return  get_correlation_matrix_diffRowsAndCols(
                rowMotifs=rowMotifs,
                colMotifs=colMotifs,
                subtracks_to_align_on=subtracks_to_align_onForCC,
                subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
                normaliseFunc=util.CROSSC_NORMFUNC.none,
                largerPerPosNormFuncs=[],
                smallerPerPosNormFuncs=[],
                account_for_rev_comp=True,
                numThreads=1,
                secondsBetweenUpdates=6,
                batch_size=batch_size,
                verbose=verbose)


def agglomerative_clustering_seqletdist(
                             motifs_list,
                             seqlets,
                             subtracks_to_align_onForCC,
                             subtracks_to_align_onForMerge,
                             sim_threshold,
                             trimming_func,
                             batch_size=20):
    original_num_motifs=len(motifs_list)
    #keeps track of which indices get merged; that's why is initialized
    #to a set
    indices_list = [set([i]) for i in range(len(motifs_list))]

    # meanAndTwoNorm used for ensuring that the corrrelations are
    # between -1 and 1
    motifs_to_seqlets_cc = get_correlation_matrix_diffRowsAndCols_shortcut(
        rowMotifs=motifs_list,
        colMotifs=seqlets,
        subtracks_to_align_onForCC=subtracks_to_align_onForCC,
        batch_size=batch_size,
        verbose=True)

    while True:
        sdevs = np.std(motifs_to_seqlets_cc, axis=1)
        sdevs = sdevs + (10**-7)*(sdevs < 10**-7)
        means = np.mean(motifs_to_seqlets_cc, axis=1)
        motifs_to_seqlets_cc =\
         (motifs_to_seqlets_cc-means[:,None])/(sdevs[:,None])
        #take correlations of the scores on things
        motifs_sim_mat = np.mean(motifs_to_seqlets_cc[:,None,:]
                                *motifs_to_seqlets_cc[None,:,:],axis=-1)

        np.fill_diagonal(motifs_sim_mat, 0)
        max_sim = np.max(motifs_sim_mat)

        print("max sim: ", max_sim, "of",len(motifs_list),"motifs")
        if max_sim < sim_threshold:
            break

        #get the pair that's most similar
        max_sim_idx1, max_sim_idx2 = \
            np.unravel_index(np.argmax(motifs_sim_mat), motifs_sim_mat.shape)

        if (motifs_list[max_sim_idx1].totalObservationsEver >=
            motifs_list[max_sim_idx2].totalObservationsEver):
            dominant_motif = motifs_list[max_sim_idx1]
            minor_motif = motifs_list[max_sim_idx2]
        else:
            dominant_motif = motifs_list[max_sim_idx2]
            minor_motif = motifs_list[max_sim_idx1]

        merged_motif = dominant_motif.merge(
            minor_motif,
            subtracks_to_align_on=subtracks_to_align_onForMerge,
            subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
            normaliseFunc=util.CROSSC_NORMFUNC.none,
            smallerPerPosNormFuncs=[],
            largerPerPosNormFuncs=[],
            revComp=True)

        removeset = {max_sim_idx1, max_sim_idx2}
        merged_indices = \
            indices_list[max_sim_idx1].union(indices_list[max_sim_idx2])
        indices_list = [indices_set
                        for i, indices_set in enumerate(indices_list)
                        if i not in removeset]
        indices_list.append(merged_indices)
        motifs_list = [motif
                         for i, motif in enumerate(motifs_list)
                         if i not in removeset]
        motifs_list.append(adjustModiscoMotifUsingTrimmingCriterion(
                              merged_motif,
                              trimming_func=trimming_func))

        #update the cross-correlation matrix for the merged motif
        mergedmotif_to_seqlets_cc =\
            get_correlation_matrix_diffRowsAndCols_shortcut(
                rowMotifs=[merged_motif],
                colMotifs=seqlets,
                subtracks_to_align_onForCC=subtracks_to_align_onForCC,
                batch_size=batch_size*original_num_motifs,
                verbose=False)
        assert len(mergedmotif_to_seqlets_cc)==1

        mask_to_keep = np.ones(len(motifs_to_seqlets_cc))
        mask_to_keep[max_sim_idx1] = 0
        mask_to_keep[max_sim_idx2] = 0
        motifs_to_seqlets_cc = np.compress(
                                    condition=mask_to_keep,
                                    a=motifs_to_seqlets_cc,
                                    axis=0)
        new_motifs_to_seqlets_cc = np.zeros((motifs_to_seqlets_cc.shape[0]+1,
                                             motifs_to_seqlets_cc.shape[1])) 
        new_motifs_to_seqlets_cc[:-1,:] = motifs_to_seqlets_cc
        new_motifs_to_seqlets_cc[-1,:] = mergedmotif_to_seqlets_cc[0] 
        motifs_to_seqlets_cc = new_motifs_to_seqlets_cc
        assert len(motifs_to_seqlets_cc)==len(motifs_list)

    return motifs_list, indices_list


def agglomerative_clustering(motifs_list,
                             cc_tracks,
                             gradient_track,
                             cc_threshold,
                             trimming_func):
    indices_list = [set([i]) for i in range(len(motifs_list))]
    while True:
        # meanAndTwoNorm used for ensuring that the corrrelations are
        # between -1 and 1
        motifs_cc = get_correlation_matrix(
            motifs_list,
            subtracks_to_align_on=cc_tracks,
            subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
            account_for_rev_comp=True,
            numThreads=1,
            secondsBetweenUpdates=6,
            batch_size=20,
            verbose=False)

        np.fill_diagonal(motifs_cc, 0)
        max_motifs_cc = np.max(motifs_cc)

        print("max cc: ", max_motifs_cc)
        if max_motifs_cc < cc_threshold:
            break

        max_cc_idx1, max_cc_idx2 = \
            np.unravel_index(np.argmax(motifs_cc), motifs_cc.shape)

        if (motifs_list[max_cc_idx1].totalObservationsEver >=
            motifs_list[max_cc_idx2].totalObservationsEver):
            dominant_motif = motifs_list[max_cc_idx1]
            minor_motif = motifs_list[max_cc_idx2]
        else:
            dominant_motif = motifs_list[max_cc_idx2]
            minor_motif = motifs_list[max_cc_idx1]

        merged_motif = dominant_motif.merge(
            minor_motif,
            subtracks_to_align_on=[gradient_track],
            subtrackNormaliseFunc=util.CROSSC_NORMFUNC.meanAndTwoNorm,
            normaliseFunc=util.CROSSC_NORMFUNC.none,
            smallerPerPosNormFuncs=[],
            largerPerPosNormFuncs=[],
            #smallerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
            #largerPerPosNormFuncs=[util.PERPOS_NORMFUNC.oneOverTwoNorm],
            revComp=True)

        removeset = {max_cc_idx1, max_cc_idx2}
        merged_indices = \
            indices_list[max_cc_idx1].union(indices_list[max_cc_idx2])
        indices_list = [indices_set
                        for i, indices_set in enumerate(indices_list)
                        if i not in removeset]
        indices_list.append(merged_indices)
        motifs_list = [motif
                         for i, motif in enumerate(motifs_list)
                         if i not in removeset]
        motifs_list.append(adjustModiscoMotifUsingTrimmingCriterion(
                              merged_motif,
                              trimming_func=trimming_func))
    return motifs_list, indices_list

def mergeAndPrintMotifs(seqlets, labels, numObsFraction
                              , mergingSubtracks
                              , subtracksToPrint
                              , printRevComp=True):
    #The trimming function is optional; it is used to further trim uninformative flanks.
    #TrimArrayColumnsToPercent trims the grammar to the smallest subsequence that has "percent" importance
    #of the original full sequence
    #trimming_func = TrimArrayColumnsToPercent(percent=0.95)
    #TrimArrayColumsnToNumUnderlyingObs resticts attention to those positions in the grammar
    #that have at least 20% of the total observations for the grammar supporting them.
    trimming_func = TrimArrayColumnsToNumUnderlyingObs(numObsFraction)
    #once again, subtracks_to_align_on indicates the subtracks to consider for merging. Should be
    #the same as what you supplied for the cross-correlation
    mergedMotifs = create_merged_modisco_motifs(labels, seqlets
                                              , subtracks_to_align_on=mergingSubtracks
                                              , account_for_rev_comp=True)
    mergedMotifs = adjust_modisco_motifs_using_trimming_criterion(mergedMotifs,trimming_func=trimming_func);
    for (motifIdx) in sorted(mergedMotifs.keys()):
        print("index",motifIdx)
        motif=mergedMotifs[motifIdx]
        print("total observations",motif.totalObservationsEver)
        print("fwd")
        printModiscoMotif(motif, trackNamesToPrint=subtracksToPrint)
        if (printRevComp):
            print("rev")
            printModiscoMotif(motif.getRevCompModiscoMotif(), trackNamesToPrint=subtracksToPrint)
    return mergedMotifs


def filter_shorter_seqlets(seqlets):
    max_seqlet_length = np.max([x.num_underlying_observations.shape[-1] for x in seqlets])
    seqlets = [x for x in seqlets if x.num_underlying_observations.shape[-1] == max_seqlet_length]
    return seqlets


def scatter_plot(xycoords, labels=None,
                colors=None, figsize=(5,5), xlabel="", ylabel=""):
    """
        If labels is not none, will assign colors using
            points evenly sampled from
            Blue -> Violet -> Red -> Yellow -> Green
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    if (labels is None):
        plt.scatter(xycoords[:,0], xycoords[:,1])
    else:
        if (colors is None):
            maxLabel = np.max(labels);
            colors = [util.fracToRainbowColour(x/float(maxLabel))
                        if x > 0 else util.fracToRainbowColour(0)
                        for x in range(maxLabel+1)];
            #print("No colors supplied, so autogen'd as:\n"+
            #        "\n".join(str(x) for x in list(enumerate(colors))))
        plt.scatter(xycoords[:,0], xycoords[:,1], c=[colors[x] for x in labels]);
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show();
