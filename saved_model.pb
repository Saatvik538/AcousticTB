§
ŃŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878đ

vggish/conv1/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namevggish/conv1/weights

(vggish/conv1/weights/Read/ReadVariableOpReadVariableOpvggish/conv1/weights*&
_output_shapes
:@*
dtype0
~
vggish/conv1/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namevggish/conv1/biases
w
'vggish/conv1/biases/Read/ReadVariableOpReadVariableOpvggish/conv1/biases*
_output_shapes
:@*
dtype0

vggish/conv2/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namevggish/conv2/weights

(vggish/conv2/weights/Read/ReadVariableOpReadVariableOpvggish/conv2/weights*'
_output_shapes
:@*
dtype0

vggish/conv2/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namevggish/conv2/biases
x
'vggish/conv2/biases/Read/ReadVariableOpReadVariableOpvggish/conv2/biases*
_output_shapes	
:*
dtype0

vggish/conv3/conv3_1/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namevggish/conv3/conv3_1/weights

0vggish/conv3/conv3_1/weights/Read/ReadVariableOpReadVariableOpvggish/conv3/conv3_1/weights*(
_output_shapes
:*
dtype0

vggish/conv3/conv3_1/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namevggish/conv3/conv3_1/biases

/vggish/conv3/conv3_1/biases/Read/ReadVariableOpReadVariableOpvggish/conv3/conv3_1/biases*
_output_shapes	
:*
dtype0

vggish/conv3/conv3_2/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namevggish/conv3/conv3_2/weights

0vggish/conv3/conv3_2/weights/Read/ReadVariableOpReadVariableOpvggish/conv3/conv3_2/weights*(
_output_shapes
:*
dtype0

vggish/conv3/conv3_2/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namevggish/conv3/conv3_2/biases

/vggish/conv3/conv3_2/biases/Read/ReadVariableOpReadVariableOpvggish/conv3/conv3_2/biases*
_output_shapes	
:*
dtype0

vggish/conv4/conv4_1/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namevggish/conv4/conv4_1/weights

0vggish/conv4/conv4_1/weights/Read/ReadVariableOpReadVariableOpvggish/conv4/conv4_1/weights*(
_output_shapes
:*
dtype0

vggish/conv4/conv4_1/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namevggish/conv4/conv4_1/biases

/vggish/conv4/conv4_1/biases/Read/ReadVariableOpReadVariableOpvggish/conv4/conv4_1/biases*
_output_shapes	
:*
dtype0

vggish/conv4/conv4_2/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namevggish/conv4/conv4_2/weights

0vggish/conv4/conv4_2/weights/Read/ReadVariableOpReadVariableOpvggish/conv4/conv4_2/weights*(
_output_shapes
:*
dtype0

vggish/conv4/conv4_2/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namevggish/conv4/conv4_2/biases

/vggish/conv4/conv4_2/biases/Read/ReadVariableOpReadVariableOpvggish/conv4/conv4_2/biases*
_output_shapes	
:*
dtype0

vggish/fc1/fc1_1/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
` *)
shared_namevggish/fc1/fc1_1/weights

,vggish/fc1/fc1_1/weights/Read/ReadVariableOpReadVariableOpvggish/fc1/fc1_1/weights* 
_output_shapes
:
` *
dtype0

vggish/fc1/fc1_1/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namevggish/fc1/fc1_1/biases

+vggish/fc1/fc1_1/biases/Read/ReadVariableOpReadVariableOpvggish/fc1/fc1_1/biases*
_output_shapes	
: *
dtype0

vggish/fc1/fc1_2/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
  *)
shared_namevggish/fc1/fc1_2/weights

,vggish/fc1/fc1_2/weights/Read/ReadVariableOpReadVariableOpvggish/fc1/fc1_2/weights* 
_output_shapes
:
  *
dtype0

vggish/fc1/fc1_2/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namevggish/fc1/fc1_2/biases

+vggish/fc1/fc1_2/biases/Read/ReadVariableOpReadVariableOpvggish/fc1/fc1_2/biases*
_output_shapes	
: *
dtype0

vggish/fc2/weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *#
shared_namevggish/fc2/weights
{
&vggish/fc2/weights/Read/ReadVariableOpReadVariableOpvggish/fc2/weights* 
_output_shapes
:
 *
dtype0
{
vggish/fc2/biasesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namevggish/fc2/biases
t
%vggish/fc2/biases/Read/ReadVariableOpReadVariableOpvggish/fc2/biases*
_output_shapes	
:*
dtype0

NoOpNoOp
§
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*â
valueŘBŐ BÎ
 

_variables

signatures

0
1
2
3
4
5
	6

7
8
9
10
11
12
13
14
15
16
17
 
QO
VARIABLE_VALUEvggish/conv1/weights'_variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEvggish/conv1/biases'_variables/1/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEvggish/conv2/weights'_variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEvggish/conv2/biases'_variables/3/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvggish/conv3/conv3_1/weights'_variables/4/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEvggish/conv3/conv3_1/biases'_variables/5/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvggish/conv3/conv3_2/weights'_variables/6/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEvggish/conv3/conv3_2/biases'_variables/7/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvggish/conv4/conv4_1/weights'_variables/8/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEvggish/conv4/conv4_1/biases'_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEvggish/conv4/conv4_2/weights(_variables/10/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvggish/conv4/conv4_2/biases(_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvggish/fc1/fc1_1/weights(_variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEvggish/fc1/fc1_1/biases(_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvggish/fc1/fc1_2/weights(_variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEvggish/fc1/fc1_2/biases(_variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEvggish/fc2/weights(_variables/16/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEvggish/fc2/biases(_variables/17/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ß
StatefulPartitionedCallStatefulPartitionedCallsaver_filename(vggish/conv1/weights/Read/ReadVariableOp'vggish/conv1/biases/Read/ReadVariableOp(vggish/conv2/weights/Read/ReadVariableOp'vggish/conv2/biases/Read/ReadVariableOp0vggish/conv3/conv3_1/weights/Read/ReadVariableOp/vggish/conv3/conv3_1/biases/Read/ReadVariableOp0vggish/conv3/conv3_2/weights/Read/ReadVariableOp/vggish/conv3/conv3_2/biases/Read/ReadVariableOp0vggish/conv4/conv4_1/weights/Read/ReadVariableOp/vggish/conv4/conv4_1/biases/Read/ReadVariableOp0vggish/conv4/conv4_2/weights/Read/ReadVariableOp/vggish/conv4/conv4_2/biases/Read/ReadVariableOp,vggish/fc1/fc1_1/weights/Read/ReadVariableOp+vggish/fc1/fc1_1/biases/Read/ReadVariableOp,vggish/fc1/fc1_2/weights/Read/ReadVariableOp+vggish/fc1/fc1_2/biases/Read/ReadVariableOp&vggish/fc2/weights/Read/ReadVariableOp%vggish/fc2/biases/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *%
f R
__inference__traced_save_688
ô
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamevggish/conv1/weightsvggish/conv1/biasesvggish/conv2/weightsvggish/conv2/biasesvggish/conv3/conv3_1/weightsvggish/conv3/conv3_1/biasesvggish/conv3/conv3_2/weightsvggish/conv3/conv3_2/biasesvggish/conv4/conv4_1/weightsvggish/conv4/conv4_1/biasesvggish/conv4/conv4_2/weightsvggish/conv4/conv4_2/biasesvggish/fc1/fc1_1/weightsvggish/fc1/fc1_1/biasesvggish/fc1/fc1_2/weightsvggish/fc1/fc1_2/biasesvggish/fc2/weightsvggish/fc2/biases*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_restore_752şŔ
ó/
ć
__inference__traced_save_688
file_prefix3
/savev2_vggish_conv1_weights_read_readvariableop2
.savev2_vggish_conv1_biases_read_readvariableop3
/savev2_vggish_conv2_weights_read_readvariableop2
.savev2_vggish_conv2_biases_read_readvariableop;
7savev2_vggish_conv3_conv3_1_weights_read_readvariableop:
6savev2_vggish_conv3_conv3_1_biases_read_readvariableop;
7savev2_vggish_conv3_conv3_2_weights_read_readvariableop:
6savev2_vggish_conv3_conv3_2_biases_read_readvariableop;
7savev2_vggish_conv4_conv4_1_weights_read_readvariableop:
6savev2_vggish_conv4_conv4_1_biases_read_readvariableop;
7savev2_vggish_conv4_conv4_2_weights_read_readvariableop:
6savev2_vggish_conv4_conv4_2_biases_read_readvariableop7
3savev2_vggish_fc1_fc1_1_weights_read_readvariableop6
2savev2_vggish_fc1_fc1_1_biases_read_readvariableop7
3savev2_vggish_fc1_fc1_2_weights_read_readvariableop6
2savev2_vggish_fc1_fc1_2_biases_read_readvariableop1
-savev2_vggish_fc2_weights_read_readvariableop0
,savev2_vggish_fc2_biases_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3ea1c01431d64a6cafd09b9fd0cc6378/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUEB(_variables/10/.ATTRIBUTES/VARIABLE_VALUEB(_variables/11/.ATTRIBUTES/VARIABLE_VALUEB(_variables/12/.ATTRIBUTES/VARIABLE_VALUEB(_variables/13/.ATTRIBUTES/VARIABLE_VALUEB(_variables/14/.ATTRIBUTES/VARIABLE_VALUEB(_variables/15/.ATTRIBUTES/VARIABLE_VALUEB(_variables/16/.ATTRIBUTES/VARIABLE_VALUEB(_variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesŽ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_vggish_conv1_weights_read_readvariableop.savev2_vggish_conv1_biases_read_readvariableop/savev2_vggish_conv2_weights_read_readvariableop.savev2_vggish_conv2_biases_read_readvariableop7savev2_vggish_conv3_conv3_1_weights_read_readvariableop6savev2_vggish_conv3_conv3_1_biases_read_readvariableop7savev2_vggish_conv3_conv3_2_weights_read_readvariableop6savev2_vggish_conv3_conv3_2_biases_read_readvariableop7savev2_vggish_conv4_conv4_1_weights_read_readvariableop6savev2_vggish_conv4_conv4_1_biases_read_readvariableop7savev2_vggish_conv4_conv4_2_weights_read_readvariableop6savev2_vggish_conv4_conv4_2_biases_read_readvariableop3savev2_vggish_fc1_fc1_1_weights_read_readvariableop2savev2_vggish_fc1_fc1_1_biases_read_readvariableop3savev2_vggish_fc1_fc1_2_weights_read_readvariableop2savev2_vggish_fc1_fc1_2_biases_read_readvariableop-savev2_vggish_fc2_weights_read_readvariableop,savev2_vggish_fc2_biases_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*đ
_input_shapesŢ
Ű: :@:@:@::::::::::
` : :
  : :
 :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
` :!

_output_shapes	
: :&"
 
_output_shapes
:
  :!

_output_shapes	
: :&"
 
_output_shapes
:
 :!

_output_shapes	
::

_output_shapes
: 
ĚM
š

__inference__traced_restore_752
file_prefix)
%assignvariableop_vggish_conv1_weights*
&assignvariableop_1_vggish_conv1_biases+
'assignvariableop_2_vggish_conv2_weights*
&assignvariableop_3_vggish_conv2_biases3
/assignvariableop_4_vggish_conv3_conv3_1_weights2
.assignvariableop_5_vggish_conv3_conv3_1_biases3
/assignvariableop_6_vggish_conv3_conv3_2_weights2
.assignvariableop_7_vggish_conv3_conv3_2_biases3
/assignvariableop_8_vggish_conv4_conv4_1_weights2
.assignvariableop_9_vggish_conv4_conv4_1_biases4
0assignvariableop_10_vggish_conv4_conv4_2_weights3
/assignvariableop_11_vggish_conv4_conv4_2_biases0
,assignvariableop_12_vggish_fc1_fc1_1_weights/
+assignvariableop_13_vggish_fc1_fc1_1_biases0
,assignvariableop_14_vggish_fc1_fc1_2_weights/
+assignvariableop_15_vggish_fc1_fc1_2_biases*
&assignvariableop_16_vggish_fc2_weights)
%assignvariableop_17_vggish_fc2_biases
identity_19˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUEB(_variables/10/.ATTRIBUTES/VARIABLE_VALUEB(_variables/11/.ATTRIBUTES/VARIABLE_VALUEB(_variables/12/.ATTRIBUTES/VARIABLE_VALUEB(_variables/13/.ATTRIBUTES/VARIABLE_VALUEB(_variables/14/.ATTRIBUTES/VARIABLE_VALUEB(_variables/15/.ATTRIBUTES/VARIABLE_VALUEB(_variables/16/.ATTRIBUTES/VARIABLE_VALUEB(_variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¤
AssignVariableOpAssignVariableOp%assignvariableop_vggish_conv1_weightsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ť
AssignVariableOp_1AssignVariableOp&assignvariableop_1_vggish_conv1_biasesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ź
AssignVariableOp_2AssignVariableOp'assignvariableop_2_vggish_conv2_weightsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ť
AssignVariableOp_3AssignVariableOp&assignvariableop_3_vggish_conv2_biasesIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4´
AssignVariableOp_4AssignVariableOp/assignvariableop_4_vggish_conv3_conv3_1_weightsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ł
AssignVariableOp_5AssignVariableOp.assignvariableop_5_vggish_conv3_conv3_1_biasesIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6´
AssignVariableOp_6AssignVariableOp/assignvariableop_6_vggish_conv3_conv3_2_weightsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ł
AssignVariableOp_7AssignVariableOp.assignvariableop_7_vggish_conv3_conv3_2_biasesIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8´
AssignVariableOp_8AssignVariableOp/assignvariableop_8_vggish_conv4_conv4_1_weightsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ł
AssignVariableOp_9AssignVariableOp.assignvariableop_9_vggish_conv4_conv4_1_biasesIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¸
AssignVariableOp_10AssignVariableOp0assignvariableop_10_vggish_conv4_conv4_2_weightsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ˇ
AssignVariableOp_11AssignVariableOp/assignvariableop_11_vggish_conv4_conv4_2_biasesIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12´
AssignVariableOp_12AssignVariableOp,assignvariableop_12_vggish_fc1_fc1_1_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ł
AssignVariableOp_13AssignVariableOp+assignvariableop_13_vggish_fc1_fc1_1_biasesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14´
AssignVariableOp_14AssignVariableOp,assignvariableop_14_vggish_fc1_fc1_2_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ł
AssignVariableOp_15AssignVariableOp+assignvariableop_15_vggish_fc1_fc1_2_biasesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ž
AssignVariableOp_16AssignVariableOp&assignvariableop_16_vggish_fc2_weightsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17­
AssignVariableOp_17AssignVariableOp%assignvariableop_17_vggish_fc2_biasesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpę
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18Ý
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ć
__inference___call___611
waveform
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallwaveformunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_wrapped_function_5692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:˙˙˙˙˙˙˙˙˙::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
waveform
×¸

 __inference_wrapped_function_569
placeholder/
+vggish_conv1_conv2d_readvariableop_resource0
,vggish_conv1_biasadd_readvariableop_resource/
+vggish_conv2_conv2d_readvariableop_resource0
,vggish_conv2_biasadd_readvariableop_resource7
3vggish_conv3_conv3_1_conv2d_readvariableop_resource8
4vggish_conv3_conv3_1_biasadd_readvariableop_resource7
3vggish_conv3_conv3_2_conv2d_readvariableop_resource8
4vggish_conv3_conv3_2_biasadd_readvariableop_resource7
3vggish_conv4_conv4_1_conv2d_readvariableop_resource8
4vggish_conv4_conv4_1_biasadd_readvariableop_resource7
3vggish_conv4_conv4_2_conv2d_readvariableop_resource8
4vggish_conv4_conv4_2_biasadd_readvariableop_resource3
/vggish_fc1_fc1_1_matmul_readvariableop_resource4
0vggish_fc1_fc1_1_biasadd_readvariableop_resource3
/vggish_fc1_fc1_2_matmul_readvariableop_resource4
0vggish_fc1_fc1_2_biasadd_readvariableop_resource-
)vggish_fc2_matmul_readvariableop_resource.
*vggish_fc2_biasadd_readvariableop_resource
vggish_embedding
"log_mel_features/stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :2$
"log_mel_features/stft/frame_length
 log_mel_features/stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B : 2"
 log_mel_features/stft/frame_step
 log_mel_features/stft/fft_lengthConst*
_output_shapes
: *
dtype0*
value
B :2"
 log_mel_features/stft/fft_length
 log_mel_features/stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2"
 log_mel_features/stft/frame/axis
!log_mel_features/stft/frame/ShapeShapeplaceholder*
T0*
_output_shapes
:2#
!log_mel_features/stft/frame/Shape
 log_mel_features/stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2"
 log_mel_features/stft/frame/Rank
'log_mel_features/stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2)
'log_mel_features/stft/frame/range/start
'log_mel_features/stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2)
'log_mel_features/stft/frame/range/deltaú
!log_mel_features/stft/frame/rangeRange0log_mel_features/stft/frame/range/start:output:0)log_mel_features/stft/frame/Rank:output:00log_mel_features/stft/frame/range/delta:output:0*
_output_shapes
:2#
!log_mel_features/stft/frame/rangeľ
/log_mel_features/stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙21
/log_mel_features/stft/frame/strided_slice/stack°
1log_mel_features/stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1log_mel_features/stft/frame/strided_slice/stack_1°
1log_mel_features/stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1log_mel_features/stft/frame/strided_slice/stack_2
)log_mel_features/stft/frame/strided_sliceStridedSlice*log_mel_features/stft/frame/range:output:08log_mel_features/stft/frame/strided_slice/stack:output:0:log_mel_features/stft/frame/strided_slice/stack_1:output:0:log_mel_features/stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)log_mel_features/stft/frame/strided_slice
!log_mel_features/stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!log_mel_features/stft/frame/sub/yÁ
log_mel_features/stft/frame/subSub)log_mel_features/stft/frame/Rank:output:0*log_mel_features/stft/frame/sub/y:output:0*
T0*
_output_shapes
: 2!
log_mel_features/stft/frame/subÇ
!log_mel_features/stft/frame/sub_1Sub#log_mel_features/stft/frame/sub:z:02log_mel_features/stft/frame/strided_slice:output:0*
T0*
_output_shapes
: 2#
!log_mel_features/stft/frame/sub_1
$log_mel_features/stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_features/stft/frame/packed/1
"log_mel_features/stft/frame/packedPack2log_mel_features/stft/frame/strided_slice:output:0-log_mel_features/stft/frame/packed/1:output:0%log_mel_features/stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2$
"log_mel_features/stft/frame/packed
+log_mel_features/stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+log_mel_features/stft/frame/split/split_dimŠ
!log_mel_features/stft/frame/splitSplitV*log_mel_features/stft/frame/Shape:output:0+log_mel_features/stft/frame/packed:output:04log_mel_features/stft/frame/split/split_dim:output:0*
T0*

Tlen0*"
_output_shapes
: :: *
	num_split2#
!log_mel_features/stft/frame/split
)log_mel_features/stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2+
)log_mel_features/stft/frame/Reshape/shapeÖ
#log_mel_features/stft/frame/ReshapeReshape*log_mel_features/stft/frame/split:output:12log_mel_features/stft/frame/Reshape/shape:output:0*
T0*
_output_shapes
: 2%
#log_mel_features/stft/frame/Reshape
 log_mel_features/stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2"
 log_mel_features/stft/frame/Size
"log_mel_features/stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"log_mel_features/stft/frame/Size_1É
!log_mel_features/stft/frame/sub_2Sub,log_mel_features/stft/frame/Reshape:output:0+log_mel_features/stft/frame_length:output:0*
T0*
_output_shapes
: 2#
!log_mel_features/stft/frame/sub_2Ë
$log_mel_features/stft/frame/floordivFloorDiv%log_mel_features/stft/frame/sub_2:z:0)log_mel_features/stft/frame_step:output:0*
T0*
_output_shapes
: 2&
$log_mel_features/stft/frame/floordiv
!log_mel_features/stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2#
!log_mel_features/stft/frame/add/xÂ
log_mel_features/stft/frame/addAddV2*log_mel_features/stft/frame/add/x:output:0(log_mel_features/stft/frame/floordiv:z:0*
T0*
_output_shapes
: 2!
log_mel_features/stft/frame/add
%log_mel_features/stft/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2'
%log_mel_features/stft/frame/Maximum/xË
#log_mel_features/stft/frame/MaximumMaximum.log_mel_features/stft/frame/Maximum/x:output:0#log_mel_features/stft/frame/add:z:0*
T0*
_output_shapes
: 2%
#log_mel_features/stft/frame/Maximum
%log_mel_features/stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :P2'
%log_mel_features/stft/frame/gcd/Const
(log_mel_features/stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :P2*
(log_mel_features/stft/frame/floordiv_1/yÝ
&log_mel_features/stft/frame/floordiv_1FloorDiv+log_mel_features/stft/frame_length:output:01log_mel_features/stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2(
&log_mel_features/stft/frame/floordiv_1
(log_mel_features/stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :P2*
(log_mel_features/stft/frame/floordiv_2/yŰ
&log_mel_features/stft/frame/floordiv_2FloorDiv)log_mel_features/stft/frame_step:output:01log_mel_features/stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2(
&log_mel_features/stft/frame/floordiv_2
(log_mel_features/stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :P2*
(log_mel_features/stft/frame/floordiv_3/yŢ
&log_mel_features/stft/frame/floordiv_3FloorDiv,log_mel_features/stft/frame/Reshape:output:01log_mel_features/stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2(
&log_mel_features/stft/frame/floordiv_3
!log_mel_features/stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :P2#
!log_mel_features/stft/frame/mul/yÂ
log_mel_features/stft/frame/mulMul*log_mel_features/stft/frame/floordiv_3:z:0*log_mel_features/stft/frame/mul/y:output:0*
T0*
_output_shapes
: 2!
log_mel_features/stft/frame/mulľ
+log_mel_features/stft/frame/concat/values_1Pack#log_mel_features/stft/frame/mul:z:0*
N*
T0*
_output_shapes
:2-
+log_mel_features/stft/frame/concat/values_1
'log_mel_features/stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'log_mel_features/stft/frame/concat/axisÂ
"log_mel_features/stft/frame/concatConcatV2*log_mel_features/stft/frame/split:output:04log_mel_features/stft/frame/concat/values_1:output:0*log_mel_features/stft/frame/split:output:20log_mel_features/stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"log_mel_features/stft/frame/concat¤
/log_mel_features/stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :P21
/log_mel_features/stft/frame/concat_1/values_1/1ú
-log_mel_features/stft/frame/concat_1/values_1Pack*log_mel_features/stft/frame/floordiv_3:z:08log_mel_features/stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2/
-log_mel_features/stft/frame/concat_1/values_1
)log_mel_features/stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)log_mel_features/stft/frame/concat_1/axisĘ
$log_mel_features/stft/frame/concat_1ConcatV2*log_mel_features/stft/frame/split:output:06log_mel_features/stft/frame/concat_1/values_1:output:0*log_mel_features/stft/frame/split:output:22log_mel_features/stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$log_mel_features/stft/frame/concat_1
&log_mel_features/stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2(
&log_mel_features/stft/frame/zeros_like¤
+log_mel_features/stft/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+log_mel_features/stft/frame/ones_like/Shape
+log_mel_features/stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+log_mel_features/stft/frame/ones_like/Constç
%log_mel_features/stft/frame/ones_likeFill4log_mel_features/stft/frame/ones_like/Shape:output:04log_mel_features/stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2'
%log_mel_features/stft/frame/ones_likeş
(log_mel_features/stft/frame/StridedSliceStridedSliceplaceholder/log_mel_features/stft/frame/zeros_like:output:0+log_mel_features/stft/frame/concat:output:0.log_mel_features/stft/frame/ones_like:output:0*
Index0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(log_mel_features/stft/frame/StridedSliceí
%log_mel_features/stft/frame/Reshape_1Reshape1log_mel_features/stft/frame/StridedSlice:output:0-log_mel_features/stft/frame/concat_1:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P2'
%log_mel_features/stft/frame/Reshape_1
)log_mel_features/stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2+
)log_mel_features/stft/frame/range_1/start
)log_mel_features/stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2+
)log_mel_features/stft/frame/range_1/delta
#log_mel_features/stft/frame/range_1Range2log_mel_features/stft/frame/range_1/start:output:0'log_mel_features/stft/frame/Maximum:z:02log_mel_features/stft/frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#log_mel_features/stft/frame/range_1Ő
!log_mel_features/stft/frame/mul_1Mul,log_mel_features/stft/frame/range_1:output:0*log_mel_features/stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!log_mel_features/stft/frame/mul_1 
-log_mel_features/stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_features/stft/frame/Reshape_2/shape/1ń
+log_mel_features/stft/frame/Reshape_2/shapePack'log_mel_features/stft/frame/Maximum:z:06log_mel_features/stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+log_mel_features/stft/frame/Reshape_2/shapeč
%log_mel_features/stft/frame/Reshape_2Reshape%log_mel_features/stft/frame/mul_1:z:04log_mel_features/stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%log_mel_features/stft/frame/Reshape_2
)log_mel_features/stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2+
)log_mel_features/stft/frame/range_2/start
)log_mel_features/stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2+
)log_mel_features/stft/frame/range_2/delta
#log_mel_features/stft/frame/range_2Range2log_mel_features/stft/frame/range_2/start:output:0*log_mel_features/stft/frame/floordiv_1:z:02log_mel_features/stft/frame/range_2/delta:output:0*
_output_shapes
:2%
#log_mel_features/stft/frame/range_2 
-log_mel_features/stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_features/stft/frame/Reshape_3/shape/0ô
+log_mel_features/stft/frame/Reshape_3/shapePack6log_mel_features/stft/frame/Reshape_3/shape/0:output:0*log_mel_features/stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2-
+log_mel_features/stft/frame/Reshape_3/shapeć
%log_mel_features/stft/frame/Reshape_3Reshape,log_mel_features/stft/frame/range_2:output:04log_mel_features/stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2'
%log_mel_features/stft/frame/Reshape_3á
!log_mel_features/stft/frame/add_1AddV2.log_mel_features/stft/frame/Reshape_2:output:0.log_mel_features/stft/frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2#
!log_mel_features/stft/frame/add_1ź
$log_mel_features/stft/frame/GatherV2GatherV2.log_mel_features/stft/frame/Reshape_1:output:0%log_mel_features/stft/frame/add_1:z:02log_mel_features/stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙P2&
$log_mel_features/stft/frame/GatherV2ę
-log_mel_features/stft/frame/concat_2/values_1Pack'log_mel_features/stft/frame/Maximum:z:0+log_mel_features/stft/frame_length:output:0*
N*
T0*
_output_shapes
:2/
-log_mel_features/stft/frame/concat_2/values_1
)log_mel_features/stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)log_mel_features/stft/frame/concat_2/axisĘ
$log_mel_features/stft/frame/concat_2ConcatV2*log_mel_features/stft/frame/split:output:06log_mel_features/stft/frame/concat_2/values_1:output:0*log_mel_features/stft/frame/split:output:22log_mel_features/stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2&
$log_mel_features/stft/frame/concat_2ę
%log_mel_features/stft/frame/Reshape_4Reshape-log_mel_features/stft/frame/GatherV2:output:0-log_mel_features/stft/frame/concat_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%log_mel_features/stft/frame/Reshape_4
*log_mel_features/stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Z2,
*log_mel_features/stft/hann_window/periodic˝
&log_mel_features/stft/hann_window/CastCast3log_mel_features/stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: 2(
&log_mel_features/stft/hann_window/Cast
,log_mel_features/stft/hann_window/FloorMod/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,log_mel_features/stft/hann_window/FloorMod/yé
*log_mel_features/stft/hann_window/FloorModFloorMod+log_mel_features/stft/frame_length:output:05log_mel_features/stft/hann_window/FloorMod/y:output:0*
T0*
_output_shapes
: 2,
*log_mel_features/stft/hann_window/FloorMod
'log_mel_features/stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'log_mel_features/stft/hann_window/sub/xŘ
%log_mel_features/stft/hann_window/subSub0log_mel_features/stft/hann_window/sub/x:output:0.log_mel_features/stft/hann_window/FloorMod:z:0*
T0*
_output_shapes
: 2'
%log_mel_features/stft/hann_window/subÍ
%log_mel_features/stft/hann_window/mulMul*log_mel_features/stft/hann_window/Cast:y:0)log_mel_features/stft/hann_window/sub:z:0*
T0*
_output_shapes
: 2'
%log_mel_features/stft/hann_window/mulĐ
%log_mel_features/stft/hann_window/addAddV2+log_mel_features/stft/frame_length:output:0)log_mel_features/stft/hann_window/mul:z:0*
T0*
_output_shapes
: 2'
%log_mel_features/stft/hann_window/add
)log_mel_features/stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2+
)log_mel_features/stft/hann_window/sub_1/yŮ
'log_mel_features/stft/hann_window/sub_1Sub)log_mel_features/stft/hann_window/add:z:02log_mel_features/stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'log_mel_features/stft/hann_window/sub_1š
(log_mel_features/stft/hann_window/Cast_1Cast+log_mel_features/stft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(log_mel_features/stft/hann_window/Cast_1 
-log_mel_features/stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-log_mel_features/stft/hann_window/range/start 
-log_mel_features/stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-log_mel_features/stft/hann_window/range/delta
'log_mel_features/stft/hann_window/rangeRange6log_mel_features/stft/hann_window/range/start:output:0+log_mel_features/stft/frame_length:output:06log_mel_features/stft/hann_window/range/delta:output:0*
_output_shapes	
:2)
'log_mel_features/stft/hann_window/rangeĂ
(log_mel_features/stft/hann_window/Cast_2Cast0log_mel_features/stft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:2*
(log_mel_features/stft/hann_window/Cast_2
'log_mel_features/stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ŰÉ@2)
'log_mel_features/stft/hann_window/Constß
'log_mel_features/stft/hann_window/mul_1Mul0log_mel_features/stft/hann_window/Const:output:0,log_mel_features/stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:2)
'log_mel_features/stft/hann_window/mul_1â
)log_mel_features/stft/hann_window/truedivRealDiv+log_mel_features/stft/hann_window/mul_1:z:0,log_mel_features/stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:2+
)log_mel_features/stft/hann_window/truedivŞ
%log_mel_features/stft/hann_window/CosCos-log_mel_features/stft/hann_window/truediv:z:0*
T0*
_output_shapes	
:2'
%log_mel_features/stft/hann_window/Cos
)log_mel_features/stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)log_mel_features/stft/hann_window/mul_2/xŢ
'log_mel_features/stft/hann_window/mul_2Mul2log_mel_features/stft/hann_window/mul_2/x:output:0)log_mel_features/stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:2)
'log_mel_features/stft/hann_window/mul_2
)log_mel_features/stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)log_mel_features/stft/hann_window/sub_2/xŕ
'log_mel_features/stft/hann_window/sub_2Sub2log_mel_features/stft/hann_window/sub_2/x:output:0+log_mel_features/stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:2)
'log_mel_features/stft/hann_window/sub_2Í
log_mel_features/stft/mulMul.log_mel_features/stft/frame/Reshape_4:output:0+log_mel_features/stft/hann_window/sub_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
log_mel_features/stft/mul§
!log_mel_features/stft/rfft/packedPack)log_mel_features/stft/fft_length:output:0*
N*
T0*
_output_shapes
:2#
!log_mel_features/stft/rfft/packedł
'log_mel_features/stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            p   2)
'log_mel_features/stft/rfft/Pad/paddingsË
log_mel_features/stft/rfft/PadPadlog_mel_features/stft/mul:z:00log_mel_features/stft/rfft/Pad/paddings:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
log_mel_features/stft/rfft/Pad
%log_mel_features/stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:2'
%log_mel_features/stft/rfft/fft_lengthĂ
log_mel_features/stft/rfftRFFT'log_mel_features/stft/rfft/Pad:output:0.log_mel_features/stft/rfft/fft_length:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
log_mel_features/stft/rfft
log_mel_features/Abs
ComplexAbs#log_mel_features/stft/rfft:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
log_mel_features/Absť
:log_mel_features/linear_to_mel_weight_matrix/sample_rate/xConst*
_output_shapes
: *
dtype0*
value
B :}2<
:log_mel_features/linear_to_mel_weight_matrix/sample_rate/xń
8log_mel_features/linear_to_mel_weight_matrix/sample_rateCastClog_mel_features/linear_to_mel_weight_matrix/sample_rate/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2:
8log_mel_features/linear_to_mel_weight_matrix/sample_rateĂ
=log_mel_features/linear_to_mel_weight_matrix/lower_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 *  úB2?
=log_mel_features/linear_to_mel_weight_matrix/lower_edge_hertzĂ
=log_mel_features/linear_to_mel_weight_matrix/upper_edge_hertzConst*
_output_shapes
: *
dtype0*
valueB
 * `ęE2?
=log_mel_features/linear_to_mel_weight_matrix/upper_edge_hertz­
2log_mel_features/linear_to_mel_weight_matrix/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2log_mel_features/linear_to_mel_weight_matrix/Constľ
6log_mel_features/linear_to_mel_weight_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @28
6log_mel_features/linear_to_mel_weight_matrix/truediv/y
4log_mel_features/linear_to_mel_weight_matrix/truedivRealDiv<log_mel_features/linear_to_mel_weight_matrix/sample_rate:y:0?log_mel_features/linear_to_mel_weight_matrix/truediv/y:output:0*
T0*
_output_shapes
: 26
4log_mel_features/linear_to_mel_weight_matrix/truedivš
9log_mel_features/linear_to_mel_weight_matrix/linspace/numConst*
_output_shapes
: *
dtype0*
value
B :2;
9log_mel_features/linear_to_mel_weight_matrix/linspace/numô
:log_mel_features/linear_to_mel_weight_matrix/linspace/CastCastBlog_mel_features/linear_to_mel_weight_matrix/linspace/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2<
:log_mel_features/linear_to_mel_weight_matrix/linspace/Castô
<log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_1Cast>log_mel_features/linear_to_mel_weight_matrix/linspace/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2>
<log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_1˝
;log_mel_features/linear_to_mel_weight_matrix/linspace/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/ShapeÁ
=log_mel_features/linear_to_mel_weight_matrix/linspace/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/Shape_1Ă
Clog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastArgsBroadcastArgsDlog_mel_features/linear_to_mel_weight_matrix/linspace/Shape:output:0Flog_mel_features/linear_to_mel_weight_matrix/linspace/Shape_1:output:0*
_output_shapes
: 2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastArgs˝
Alog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastToBroadcastTo;log_mel_features/linear_to_mel_weight_matrix/Const:output:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 2C
Alog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastTož
Clog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastTo_1BroadcastTo8log_mel_features/linear_to_mel_weight_matrix/truediv:z:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastArgs:r0:0*
T0*
_output_shapes
: 2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastTo_1Î
Dlog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dlog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims/dimŇ
@log_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims
ExpandDimsJlog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastTo:output:0Mlog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims/dim:output:0*
T0*
_output_shapes
:2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDimsŇ
Flog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Flog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1/dimÚ
Blog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1
ExpandDimsLlog_mel_features/linear_to_mel_weight_matrix/linspace/BroadcastTo_1:output:0Olog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1Č
=log_mel_features/linear_to_mel_weight_matrix/linspace/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/Shape_2Č
=log_mel_features/linear_to_mel_weight_matrix/linspace/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/Shape_3ŕ
Ilog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Ilog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stackä
Klog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Klog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack_1ä
Klog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Klog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack_2¨
Clog_mel_features/linear_to_mel_weight_matrix/linspace/strided_sliceStridedSliceFlog_mel_features/linear_to_mel_weight_matrix/linspace/Shape_3:output:0Rlog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack:output:0Tlog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack_1:output:0Tlog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace/strided_sliceź
;log_mel_features/linear_to_mel_weight_matrix/linspace/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/add/y´
9log_mel_features/linear_to_mel_weight_matrix/linspace/addAddV2Llog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice:output:0Dlog_mel_features/linear_to_mel_weight_matrix/linspace/add/y:output:0*
T0*
_output_shapes
: 2;
9log_mel_features/linear_to_mel_weight_matrix/linspace/addÖ
Hlog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z2J
Hlog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2/conditionĆ
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2/t
>log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2SelectV2Qlog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2/condition:output:0Ilog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2/t:output:0=log_mel_features/linear_to_mel_weight_matrix/linspace/add:z:0*
T0*
_output_shapes
: 2@
>log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2ź
;log_mel_features/linear_to_mel_weight_matrix/linspace/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/sub/y¤
9log_mel_features/linear_to_mel_weight_matrix/linspace/subSub>log_mel_features/linear_to_mel_weight_matrix/linspace/Cast:y:0Dlog_mel_features/linear_to_mel_weight_matrix/linspace/sub/y:output:0*
T0*
_output_shapes
: 2;
9log_mel_features/linear_to_mel_weight_matrix/linspace/subÄ
?log_mel_features/linear_to_mel_weight_matrix/linspace/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 2A
?log_mel_features/linear_to_mel_weight_matrix/linspace/Maximum/ył
=log_mel_features/linear_to_mel_weight_matrix/linspace/MaximumMaximum=log_mel_features/linear_to_mel_weight_matrix/linspace/sub:z:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum/y:output:0*
T0*
_output_shapes
: 2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/MaximumŔ
=log_mel_features/linear_to_mel_weight_matrix/linspace/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/sub_1/yŞ
;log_mel_features/linear_to_mel_weight_matrix/linspace/sub_1Sub>log_mel_features/linear_to_mel_weight_matrix/linspace/Cast:y:0Flog_mel_features/linear_to_mel_weight_matrix/linspace/sub_1/y:output:0*
T0*
_output_shapes
: 2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/sub_1Č
Alog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :2C
Alog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1/yť
?log_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1Maximum?log_mel_features/linear_to_mel_weight_matrix/linspace/sub_1:z:0Jlog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1/y:output:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1ž
;log_mel_features/linear_to_mel_weight_matrix/linspace/sub_2SubKlog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:0Ilog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims:output:0*
T0*
_output_shapes
:2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/sub_2ů
<log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_2CastClog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2>
<log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_2ą
=log_mel_features/linear_to_mel_weight_matrix/linspace/truedivRealDiv?log_mel_features/linear_to_mel_weight_matrix/linspace/sub_2:z:0@log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_2:y:0*
T0*
_output_shapes
:2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/truedivÎ
Dlog_mel_features/linear_to_mel_weight_matrix/linspace/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dlog_mel_features/linear_to_mel_weight_matrix/linspace/GreaterEqual/yČ
Blog_mel_features/linear_to_mel_weight_matrix/linspace/GreaterEqualGreaterEqual>log_mel_features/linear_to_mel_weight_matrix/linspace/Cast:y:0Mlog_mel_features/linear_to_mel_weight_matrix/linspace/GreaterEqual/y:output:0*
T0*
_output_shapes
: 2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace/GreaterEqualÓ
Blog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_1/e
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_1SelectV2Flog_mel_features/linear_to_mel_weight_matrix/linspace/GreaterEqual:z:0Clog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum_1:z:0Klog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_1/e:output:0*
T0*
_output_shapes
: 2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_1Č
Alog_mel_features/linear_to_mel_weight_matrix/linspace/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R2C
Alog_mel_features/linear_to_mel_weight_matrix/linspace/range/startČ
Alog_mel_features/linear_to_mel_weight_matrix/linspace/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2C
Alog_mel_features/linear_to_mel_weight_matrix/linspace/range/delta
@log_mel_features/linear_to_mel_weight_matrix/linspace/range/CastCastIlog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/range/Cast
;log_mel_features/linear_to_mel_weight_matrix/linspace/rangeRangeJlog_mel_features/linear_to_mel_weight_matrix/linspace/range/start:output:0Dlog_mel_features/linear_to_mel_weight_matrix/linspace/range/Cast:y:0Jlog_mel_features/linear_to_mel_weight_matrix/linspace/range/delta:output:0*

Tidx0	*
_output_shapes	
:˙2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/range˙
<log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_3CastDlog_mel_features/linear_to_mel_weight_matrix/linspace/range:output:0*

DstT0*

SrcT0	*
_output_shapes	
:˙2>
<log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_3Ě
Clog_mel_features/linear_to_mel_weight_matrix/linspace/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace/range_1/startĚ
Clog_mel_features/linear_to_mel_weight_matrix/linspace/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace/range_1/delta
=log_mel_features/linear_to_mel_weight_matrix/linspace/range_1RangeLlog_mel_features/linear_to_mel_weight_matrix/linspace/range_1/start:output:0Llog_mel_features/linear_to_mel_weight_matrix/linspace/strided_slice:output:0Llog_mel_features/linear_to_mel_weight_matrix/linspace/range_1/delta:output:0*
_output_shapes
:2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/range_1š
;log_mel_features/linear_to_mel_weight_matrix/linspace/EqualEqualGlog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2:output:0Flog_mel_features/linear_to_mel_weight_matrix/linspace/range_1:output:0*
T0*
_output_shapes
:2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/EqualĘ
Blog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_2/e
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_2SelectV2?log_mel_features/linear_to_mel_weight_matrix/linspace/Equal:z:0Alog_mel_features/linear_to_mel_weight_matrix/linspace/Maximum:z:0Klog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_2/e:output:0*
T0*
_output_shapes
:2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_2ź
=log_mel_features/linear_to_mel_weight_matrix/linspace/ReshapeReshape@log_mel_features/linear_to_mel_weight_matrix/linspace/Cast_3:y:0Ilog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_2:output:0*
T0*
_output_shapes	
:˙2?
=log_mel_features/linear_to_mel_weight_matrix/linspace/ReshapeŽ
9log_mel_features/linear_to_mel_weight_matrix/linspace/mulMulAlog_mel_features/linear_to_mel_weight_matrix/linspace/truediv:z:0Flog_mel_features/linear_to_mel_weight_matrix/linspace/Reshape:output:0*
T0*
_output_shapes	
:˙2;
9log_mel_features/linear_to_mel_weight_matrix/linspace/mulł
;log_mel_features/linear_to_mel_weight_matrix/linspace/add_1AddV2Ilog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims:output:0=log_mel_features/linear_to_mel_weight_matrix/linspace/mul:z:0*
T0*
_output_shapes	
:˙2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/add_1Ů
<log_mel_features/linear_to_mel_weight_matrix/linspace/concatConcatV2Ilog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims:output:0?log_mel_features/linear_to_mel_weight_matrix/linspace/add_1:z:0Klog_mel_features/linear_to_mel_weight_matrix/linspace/ExpandDims_1:output:0Glog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2:output:0*
N*
T0*
_output_shapes	
:2>
<log_mel_features/linear_to_mel_weight_matrix/linspace/concatÎ
@log_mel_features/linear_to_mel_weight_matrix/linspace/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/zeros_likeţ
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_3SelectV2?log_mel_features/linear_to_mel_weight_matrix/linspace/Equal:z:0>log_mel_features/linear_to_mel_weight_matrix/linspace/Cast:y:0Flog_mel_features/linear_to_mel_weight_matrix/linspace/Shape_2:output:0*
T0*
_output_shapes
:2B
@log_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_3
;log_mel_features/linear_to_mel_weight_matrix/linspace/SliceSliceElog_mel_features/linear_to_mel_weight_matrix/linspace/concat:output:0Ilog_mel_features/linear_to_mel_weight_matrix/linspace/zeros_like:output:0Ilog_mel_features/linear_to_mel_weight_matrix/linspace/SelectV2_3:output:0*
Index0*
T0*
_output_shapes	
:2=
;log_mel_features/linear_to_mel_weight_matrix/linspace/SliceÎ
@log_mel_features/linear_to_mel_weight_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@log_mel_features/linear_to_mel_weight_matrix/strided_slice/stackŇ
Blog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Blog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack_1Ň
Blog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Blog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack_2ö
:log_mel_features/linear_to_mel_weight_matrix/strided_sliceStridedSliceDlog_mel_features/linear_to_mel_weight_matrix/linspace/Slice:output:0Ilog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack:output:0Klog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack_1:output:0Klog_mel_features/linear_to_mel_weight_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes	
:*
end_mask2<
:log_mel_features/linear_to_mel_weight_matrix/strided_sliceĎ
Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D2E
Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/truediv/yĘ
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/truedivRealDivClog_mel_features/linear_to_mel_weight_matrix/strided_slice:output:0Llog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/truediv/y:output:0*
T0*
_output_shapes	
:2C
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/truedivÇ
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/add/xž
=log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/addAddV2Hlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/add/x:output:0Elog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/truediv:z:0*
T0*
_output_shapes	
:2?
=log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/addî
=log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/LogLogAlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/add:z:0*
T0*
_output_shapes	
:2?
=log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/LogÇ
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ŕD2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/mul/x¸
=log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/mulMulHlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/mul/x:output:0Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/Log:y:0*
T0*
_output_shapes	
:2?
=log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/mulź
;log_mel_features/linear_to_mel_weight_matrix/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;log_mel_features/linear_to_mel_weight_matrix/ExpandDims/dimł
7log_mel_features/linear_to_mel_weight_matrix/ExpandDims
ExpandDimsAlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel/mul:z:0Dlog_mel_features/linear_to_mel_weight_matrix/ExpandDims/dim:output:0*
T0*
_output_shapes
:	29
7log_mel_features/linear_to_mel_weight_matrix/ExpandDimsÓ
Elog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D2G
Elog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/yÎ
Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/truedivRealDivFlog_mel_features/linear_to_mel_weight_matrix/lower_edge_hertz:output:0Nlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/truediv/y:output:0*
T0*
_output_shapes
: 2E
Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/truedivË
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/add/xÁ
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/addAddV2Jlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/add/x:output:0Glog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/truediv:z:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/addď
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/LogLogClog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/add:z:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/LogË
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ŕD2C
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/mul/xť
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/mulMulJlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/mul/x:output:0Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/Log:y:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/mulÓ
Elog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  /D2G
Elog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/yÎ
Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/truedivRealDivFlog_mel_features/linear_to_mel_weight_matrix/upper_edge_hertz:output:0Nlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/truediv/y:output:0*
T0*
_output_shapes
: 2E
Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/truedivË
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/add/xÁ
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/addAddV2Jlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/add/x:output:0Glog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/truediv:z:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/addď
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/LogLogClog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/add:z:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/LogË
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 * ŕD2C
Alog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/mul/xť
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/mulMulJlog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/mul/x:output:0Clog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/Log:y:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/mulź
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/numConst*
_output_shapes
: *
dtype0*
value	B :B2=
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/numú
<log_mel_features/linear_to_mel_weight_matrix/linspace_1/CastCastDlog_mel_features/linear_to_mel_weight_matrix/linspace_1/num:output:0*

DstT0*

SrcT0*
_output_shapes
: 2>
<log_mel_features/linear_to_mel_weight_matrix/linspace_1/Castú
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_1Cast@log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast:y:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_1Á
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/ShapeĹ
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_1Ë
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastArgsBroadcastArgsFlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape:output:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_1:output:0*
_output_shapes
: 2G
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastArgsË
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastToBroadcastToClog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_1/mul:z:0Jlog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastToĎ
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1BroadcastToClog_mel_features/linear_to_mel_weight_matrix/hertz_to_mel_2/mul:z:0Jlog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastArgs:r0:0*
T0*
_output_shapes
: 2G
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1Ň
Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims/dimÚ
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims
ExpandDimsLlog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastTo:output:0Olog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDimsÖ
Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dimâ
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1
ExpandDimsNlog_mel_features/linear_to_mel_weight_matrix/linspace_1/BroadcastTo_1:output:0Qlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:2F
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1Ě
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_2Ě
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_3ä
Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stackč
Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1č
Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2´
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_sliceStridedSliceHlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_3:output:0Tlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack:output:0Vlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_1:output:0Vlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_sliceŔ
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/add/yź
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/addAddV2Nlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/add/y:output:0*
T0*
_output_shapes
: 2=
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/addÚ
Jlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z2L
Jlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2/conditionĘ
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
value	B : 2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2/t
@log_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2SelectV2Slog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2/condition:output:0Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2/t:output:0?log_mel_features/linear_to_mel_weight_matrix/linspace_1/add:z:0*
T0*
_output_shapes
: 2B
@log_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2Ŕ
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub/yŹ
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/subSub@log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast:y:0Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/sub/y:output:0*
T0*
_output_shapes
: 2=
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/subČ
Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B : 2C
Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum/yť
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/MaximumMaximum?log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub:z:0Jlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum/y:output:0*
T0*
_output_shapes
: 2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/MaximumÄ
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_1/y˛
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_1Sub@log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast:y:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_1/y:output:0*
T0*
_output_shapes
: 2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_1Ě
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1/yĂ
Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1MaximumAlog_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_1:z:0Llog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1/y:output:0*
T0*
_output_shapes
: 2C
Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1Ć
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_2SubMlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0*
T0*
_output_shapes
:2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_2˙
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_2CastElog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_2š
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/truedivRealDivAlog_mel_features/linear_to_mel_weight_matrix/linspace_1/sub_2:z:0Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_2:y:0*
T0*
_output_shapes
:2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/truedivŇ
Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2H
Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/GreaterEqual/yĐ
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/GreaterEqualGreaterEqual@log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast:y:0Olog_mel_features/linear_to_mel_weight_matrix/linspace_1/GreaterEqual/y:output:0*
T0*
_output_shapes
: 2F
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/GreaterEqual×
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_1/eConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙2F
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_1SelectV2Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/GreaterEqual:z:0Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum_1:z:0Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_1/e:output:0*
T0*
_output_shapes
: 2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_1Ě
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/startĚ
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R2E
Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/delta
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/CastCastKlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/Cast
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/rangeRangeLlog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/start:output:0Flog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/Cast:y:0Llog_mel_features/linear_to_mel_weight_matrix/linspace_1/range/delta:output:0*

Tidx0	*
_output_shapes
:@2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/range
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_3CastFlog_mel_features/linear_to_mel_weight_matrix/linspace_1/range:output:0*

DstT0*

SrcT0	*
_output_shapes
:@2@
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_3Đ
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2G
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1/startĐ
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2G
Elog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1/delta
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1RangeNlog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1/start:output:0Nlog_mel_features/linear_to_mel_weight_matrix/linspace_1/strided_slice:output:0Nlog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1/delta:output:0*
_output_shapes
:2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1Á
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/EqualEqualIlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/range_1:output:0*
T0*
_output_shapes
:2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/EqualÎ
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_2/eConst*
_output_shapes
: *
dtype0*
value	B :2F
Dlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_2SelectV2Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/Equal:z:0Clog_mel_features/linear_to_mel_weight_matrix/linspace_1/Maximum:z:0Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_2/e:output:0*
T0*
_output_shapes
:2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_2Ă
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/ReshapeReshapeBlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast_3:y:0Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_2:output:0*
T0*
_output_shapes
:@2A
?log_mel_features/linear_to_mel_weight_matrix/linspace_1/Reshapeľ
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/mulMulClog_mel_features/linear_to_mel_weight_matrix/linspace_1/truediv:z:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Reshape:output:0*
T0*
_output_shapes
:@2=
;log_mel_features/linear_to_mel_weight_matrix/linspace_1/mulş
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/add_1AddV2Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0?log_mel_features/linear_to_mel_weight_matrix/linspace_1/mul:z:0*
T0*
_output_shapes
:@2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/add_1ä
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/concatConcatV2Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims:output:0Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/add_1:z:0Mlog_mel_features/linear_to_mel_weight_matrix/linspace_1/ExpandDims_1:output:0Ilog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2:output:0*
N*
T0*
_output_shapes
:B2@
>log_mel_features/linear_to_mel_weight_matrix/linspace_1/concatŇ
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/zeros_like
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_3SelectV2Alog_mel_features/linear_to_mel_weight_matrix/linspace_1/Equal:z:0@log_mel_features/linear_to_mel_weight_matrix/linspace_1/Cast:y:0Hlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Shape_2:output:0*
T0*
_output_shapes
:2D
Blog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_3
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/SliceSliceGlog_mel_features/linear_to_mel_weight_matrix/linspace_1/concat:output:0Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/zeros_like:output:0Klog_mel_features/linear_to_mel_weight_matrix/linspace_1/SelectV2_3:output:0*
Index0*
T0*
_output_shapes
:B2?
=log_mel_features/linear_to_mel_weight_matrix/linspace_1/SliceÄ
?log_mel_features/linear_to_mel_weight_matrix/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :2A
?log_mel_features/linear_to_mel_weight_matrix/frame/frame_lengthŔ
=log_mel_features/linear_to_mel_weight_matrix/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :2?
=log_mel_features/linear_to_mel_weight_matrix/frame/frame_step˝
7log_mel_features/linear_to_mel_weight_matrix/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙29
7log_mel_features/linear_to_mel_weight_matrix/frame/axisž
8log_mel_features/linear_to_mel_weight_matrix/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB:B2:
8log_mel_features/linear_to_mel_weight_matrix/frame/ShapeÁ
=log_mel_features/linear_to_mel_weight_matrix/frame/Size/ConstConst*
_output_shapes
: *
dtype0*
valueB 2?
=log_mel_features/linear_to_mel_weight_matrix/frame/Size/Const´
7log_mel_features/linear_to_mel_weight_matrix/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 29
7log_mel_features/linear_to_mel_weight_matrix/frame/SizeĹ
?log_mel_features/linear_to_mel_weight_matrix/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2A
?log_mel_features/linear_to_mel_weight_matrix/frame/Size_1/Const¸
9log_mel_features/linear_to_mel_weight_matrix/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : 2;
9log_mel_features/linear_to_mel_weight_matrix/frame/Size_1ś
8log_mel_features/linear_to_mel_weight_matrix/frame/sub/xConst*
_output_shapes
: *
dtype0*
value	B :B2:
8log_mel_features/linear_to_mel_weight_matrix/frame/sub/xĽ
6log_mel_features/linear_to_mel_weight_matrix/frame/subSubAlog_mel_features/linear_to_mel_weight_matrix/frame/sub/x:output:0Hlog_mel_features/linear_to_mel_weight_matrix/frame/frame_length:output:0*
T0*
_output_shapes
: 28
6log_mel_features/linear_to_mel_weight_matrix/frame/subŤ
;log_mel_features/linear_to_mel_weight_matrix/frame/floordivFloorDiv:log_mel_features/linear_to_mel_weight_matrix/frame/sub:z:0Flog_mel_features/linear_to_mel_weight_matrix/frame/frame_step:output:0*
T0*
_output_shapes
: 2=
;log_mel_features/linear_to_mel_weight_matrix/frame/floordivś
8log_mel_features/linear_to_mel_weight_matrix/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2:
8log_mel_features/linear_to_mel_weight_matrix/frame/add/x
6log_mel_features/linear_to_mel_weight_matrix/frame/addAddV2Alog_mel_features/linear_to_mel_weight_matrix/frame/add/x:output:0?log_mel_features/linear_to_mel_weight_matrix/frame/floordiv:z:0*
T0*
_output_shapes
: 28
6log_mel_features/linear_to_mel_weight_matrix/frame/addž
<log_mel_features/linear_to_mel_weight_matrix/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2>
<log_mel_features/linear_to_mel_weight_matrix/frame/Maximum/x§
:log_mel_features/linear_to_mel_weight_matrix/frame/MaximumMaximumElog_mel_features/linear_to_mel_weight_matrix/frame/Maximum/x:output:0:log_mel_features/linear_to_mel_weight_matrix/frame/add:z:0*
T0*
_output_shapes
: 2<
:log_mel_features/linear_to_mel_weight_matrix/frame/Maximumž
<log_mel_features/linear_to_mel_weight_matrix/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :2>
<log_mel_features/linear_to_mel_weight_matrix/frame/gcd/ConstÄ
?log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1/yż
=log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1FloorDivHlog_mel_features/linear_to_mel_weight_matrix/frame/frame_length:output:0Hlog_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2?
=log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1Ä
?log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :2A
?log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_2/y˝
=log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_2FloorDivFlog_mel_features/linear_to_mel_weight_matrix/frame/frame_step:output:0Hlog_mel_features/linear_to_mel_weight_matrix/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2?
=log_mel_features/linear_to_mel_weight_matrix/frame/floordiv_2Ë
Blog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_0Const*
_output_shapes
: *
dtype0*
valueB 2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_0Ň
Blog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:B2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_1Ë
Blog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB 2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_2Â
>log_mel_features/linear_to_mel_weight_matrix/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>log_mel_features/linear_to_mel_weight_matrix/frame/concat/axisŕ
9log_mel_features/linear_to_mel_weight_matrix/frame/concatConcatV2Klog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_0:output:0Klog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_1:output:0Klog_mel_features/linear_to_mel_weight_matrix/frame/concat/values_2:output:0Glog_mel_features/linear_to_mel_weight_matrix/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9log_mel_features/linear_to_mel_weight_matrix/frame/concatĎ
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_0Const*
_output_shapes
: *
dtype0*
valueB 2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_0Ý
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"B      2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_1Ď
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB 2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_2Ć
@log_mel_features/linear_to_mel_weight_matrix/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@log_mel_features/linear_to_mel_weight_matrix/frame/concat_1/axisě
;log_mel_features/linear_to_mel_weight_matrix/frame/concat_1ConcatV2Mlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_0:output:0Mlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_1:output:0Mlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/values_2:output:0Ilog_mel_features/linear_to_mel_weight_matrix/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;log_mel_features/linear_to_mel_weight_matrix/frame/concat_1Ö
Dlog_mel_features/linear_to_mel_weight_matrix/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB:B2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/zeros_like/tensorČ
=log_mel_features/linear_to_mel_weight_matrix/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2?
=log_mel_features/linear_to_mel_weight_matrix/frame/zeros_likeŇ
Blog_mel_features/linear_to_mel_weight_matrix/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/ones_like/ShapeĘ
Blog_mel_features/linear_to_mel_weight_matrix/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/ones_like/ConstĂ
<log_mel_features/linear_to_mel_weight_matrix/frame/ones_likeFillKlog_mel_features/linear_to_mel_weight_matrix/frame/ones_like/Shape:output:0Klog_mel_features/linear_to_mel_weight_matrix/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2>
<log_mel_features/linear_to_mel_weight_matrix/frame/ones_likeß
?log_mel_features/linear_to_mel_weight_matrix/frame/StridedSliceStridedSliceFlog_mel_features/linear_to_mel_weight_matrix/linspace_1/Slice:output:0Flog_mel_features/linear_to_mel_weight_matrix/frame/zeros_like:output:0Blog_mel_features/linear_to_mel_weight_matrix/frame/concat:output:0Elog_mel_features/linear_to_mel_weight_matrix/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:B2A
?log_mel_features/linear_to_mel_weight_matrix/frame/StridedSliceź
:log_mel_features/linear_to_mel_weight_matrix/frame/ReshapeReshapeHlog_mel_features/linear_to_mel_weight_matrix/frame/StridedSlice:output:0Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_1:output:0*
T0*
_output_shapes

:B2<
:log_mel_features/linear_to_mel_weight_matrix/frame/ReshapeÂ
>log_mel_features/linear_to_mel_weight_matrix/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2@
>log_mel_features/linear_to_mel_weight_matrix/frame/range/startÂ
>log_mel_features/linear_to_mel_weight_matrix/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2@
>log_mel_features/linear_to_mel_weight_matrix/frame/range/deltaë
8log_mel_features/linear_to_mel_weight_matrix/frame/rangeRangeGlog_mel_features/linear_to_mel_weight_matrix/frame/range/start:output:0>log_mel_features/linear_to_mel_weight_matrix/frame/Maximum:z:0Glog_mel_features/linear_to_mel_weight_matrix/frame/range/delta:output:0*
_output_shapes
:@2:
8log_mel_features/linear_to_mel_weight_matrix/frame/range˘
6log_mel_features/linear_to_mel_weight_matrix/frame/mulMulAlog_mel_features/linear_to_mel_weight_matrix/frame/range:output:0Alog_mel_features/linear_to_mel_weight_matrix/frame/floordiv_2:z:0*
T0*
_output_shapes
:@28
6log_mel_features/linear_to_mel_weight_matrix/frame/mulÎ
Dlog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1/shape/1Í
Blog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1/shapePack>log_mel_features/linear_to_mel_weight_matrix/frame/Maximum:z:0Mlog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1/shapeš
<log_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1Reshape:log_mel_features/linear_to_mel_weight_matrix/frame/mul:z:0Klog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2>
<log_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1Ć
@log_mel_features/linear_to_mel_weight_matrix/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2B
@log_mel_features/linear_to_mel_weight_matrix/frame/range_1/startĆ
@log_mel_features/linear_to_mel_weight_matrix/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2B
@log_mel_features/linear_to_mel_weight_matrix/frame/range_1/deltaö
:log_mel_features/linear_to_mel_weight_matrix/frame/range_1RangeIlog_mel_features/linear_to_mel_weight_matrix/frame/range_1/start:output:0Alog_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1:z:0Ilog_mel_features/linear_to_mel_weight_matrix/frame/range_1/delta:output:0*
_output_shapes
:2<
:log_mel_features/linear_to_mel_weight_matrix/frame/range_1Î
Dlog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2/shape/0Đ
Blog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2/shapePackMlog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2/shape/0:output:0Alog_mel_features/linear_to_mel_weight_matrix/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2D
Blog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2/shapeÂ
<log_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2ReshapeClog_mel_features/linear_to_mel_weight_matrix/frame/range_1:output:0Klog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2/shape:output:0*
T0*
_output_shapes

:2>
<log_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2´
8log_mel_features/linear_to_mel_weight_matrix/frame/add_1AddV2Elog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_1:output:0Elog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_2:output:0*
T0*
_output_shapes

:@2:
8log_mel_features/linear_to_mel_weight_matrix/frame/add_1Ć
@log_mel_features/linear_to_mel_weight_matrix/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@log_mel_features/linear_to_mel_weight_matrix/frame/GatherV2/axis¤
;log_mel_features/linear_to_mel_weight_matrix/frame/GatherV2GatherV2Clog_mel_features/linear_to_mel_weight_matrix/frame/Reshape:output:0<log_mel_features/linear_to_mel_weight_matrix/frame/add_1:z:0Ilog_mel_features/linear_to_mel_weight_matrix/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*"
_output_shapes
:@2=
;log_mel_features/linear_to_mel_weight_matrix/frame/GatherV2Ď
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_0Const*
_output_shapes
: *
dtype0*
valueB 2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_0Ě
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_1Pack>log_mel_features/linear_to_mel_weight_matrix/frame/Maximum:z:0Hlog_mel_features/linear_to_mel_weight_matrix/frame/frame_length:output:0*
N*
T0*
_output_shapes
:2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_1Ď
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_2Const*
_output_shapes
: *
dtype0*
valueB 2F
Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_2Ć
@log_mel_features/linear_to_mel_weight_matrix/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@log_mel_features/linear_to_mel_weight_matrix/frame/concat_2/axisě
;log_mel_features/linear_to_mel_weight_matrix/frame/concat_2ConcatV2Mlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_0:output:0Mlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_1:output:0Mlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/values_2:output:0Ilog_mel_features/linear_to_mel_weight_matrix/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2=
;log_mel_features/linear_to_mel_weight_matrix/frame/concat_2ź
<log_mel_features/linear_to_mel_weight_matrix/frame/Reshape_3ReshapeDlog_mel_features/linear_to_mel_weight_matrix/frame/GatherV2:output:0Dlog_mel_features/linear_to_mel_weight_matrix/frame/concat_2:output:0*
T0*
_output_shapes

:@2>
<log_mel_features/linear_to_mel_weight_matrix/frame/Reshape_3Ž
4log_mel_features/linear_to_mel_weight_matrix/Const_1Const*
_output_shapes
: *
dtype0*
value	B :26
4log_mel_features/linear_to_mel_weight_matrix/Const_1ž
<log_mel_features/linear_to_mel_weight_matrix/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2>
<log_mel_features/linear_to_mel_weight_matrix/split/split_dimÍ
2log_mel_features/linear_to_mel_weight_matrix/splitSplitElog_mel_features/linear_to_mel_weight_matrix/split/split_dim:output:0Elog_mel_features/linear_to_mel_weight_matrix/frame/Reshape_3:output:0*
T0*2
_output_shapes 
:@:@:@*
	num_split24
2log_mel_features/linear_to_mel_weight_matrix/splitÉ
:log_mel_features/linear_to_mel_weight_matrix/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2<
:log_mel_features/linear_to_mel_weight_matrix/Reshape/shape˘
4log_mel_features/linear_to_mel_weight_matrix/ReshapeReshape;log_mel_features/linear_to_mel_weight_matrix/split:output:0Clog_mel_features/linear_to_mel_weight_matrix/Reshape/shape:output:0*
T0*
_output_shapes

:@26
4log_mel_features/linear_to_mel_weight_matrix/ReshapeÍ
<log_mel_features/linear_to_mel_weight_matrix/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2>
<log_mel_features/linear_to_mel_weight_matrix/Reshape_1/shape¨
6log_mel_features/linear_to_mel_weight_matrix/Reshape_1Reshape;log_mel_features/linear_to_mel_weight_matrix/split:output:1Elog_mel_features/linear_to_mel_weight_matrix/Reshape_1/shape:output:0*
T0*
_output_shapes

:@28
6log_mel_features/linear_to_mel_weight_matrix/Reshape_1Í
<log_mel_features/linear_to_mel_weight_matrix/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2>
<log_mel_features/linear_to_mel_weight_matrix/Reshape_2/shape¨
6log_mel_features/linear_to_mel_weight_matrix/Reshape_2Reshape;log_mel_features/linear_to_mel_weight_matrix/split:output:2Elog_mel_features/linear_to_mel_weight_matrix/Reshape_2/shape:output:0*
T0*
_output_shapes

:@28
6log_mel_features/linear_to_mel_weight_matrix/Reshape_2
0log_mel_features/linear_to_mel_weight_matrix/subSub@log_mel_features/linear_to_mel_weight_matrix/ExpandDims:output:0=log_mel_features/linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes
:	@22
0log_mel_features/linear_to_mel_weight_matrix/sub
2log_mel_features/linear_to_mel_weight_matrix/sub_1Sub?log_mel_features/linear_to_mel_weight_matrix/Reshape_1:output:0=log_mel_features/linear_to_mel_weight_matrix/Reshape:output:0*
T0*
_output_shapes

:@24
2log_mel_features/linear_to_mel_weight_matrix/sub_1
6log_mel_features/linear_to_mel_weight_matrix/truediv_1RealDiv4log_mel_features/linear_to_mel_weight_matrix/sub:z:06log_mel_features/linear_to_mel_weight_matrix/sub_1:z:0*
T0*
_output_shapes
:	@28
6log_mel_features/linear_to_mel_weight_matrix/truediv_1
2log_mel_features/linear_to_mel_weight_matrix/sub_2Sub?log_mel_features/linear_to_mel_weight_matrix/Reshape_2:output:0@log_mel_features/linear_to_mel_weight_matrix/ExpandDims:output:0*
T0*
_output_shapes
:	@24
2log_mel_features/linear_to_mel_weight_matrix/sub_2
2log_mel_features/linear_to_mel_weight_matrix/sub_3Sub?log_mel_features/linear_to_mel_weight_matrix/Reshape_2:output:0?log_mel_features/linear_to_mel_weight_matrix/Reshape_1:output:0*
T0*
_output_shapes

:@24
2log_mel_features/linear_to_mel_weight_matrix/sub_3
6log_mel_features/linear_to_mel_weight_matrix/truediv_2RealDiv6log_mel_features/linear_to_mel_weight_matrix/sub_2:z:06log_mel_features/linear_to_mel_weight_matrix/sub_3:z:0*
T0*
_output_shapes
:	@28
6log_mel_features/linear_to_mel_weight_matrix/truediv_2
4log_mel_features/linear_to_mel_weight_matrix/MinimumMinimum:log_mel_features/linear_to_mel_weight_matrix/truediv_1:z:0:log_mel_features/linear_to_mel_weight_matrix/truediv_2:z:0*
T0*
_output_shapes
:	@26
4log_mel_features/linear_to_mel_weight_matrix/Minimum
4log_mel_features/linear_to_mel_weight_matrix/MaximumMaximum;log_mel_features/linear_to_mel_weight_matrix/Const:output:08log_mel_features/linear_to_mel_weight_matrix/Minimum:z:0*
T0*
_output_shapes
:	@26
4log_mel_features/linear_to_mel_weight_matrix/MaximumĎ
5log_mel_features/linear_to_mel_weight_matrix/paddingsConst*
_output_shapes

:*
dtype0*)
value B"               27
5log_mel_features/linear_to_mel_weight_matrix/paddings
,log_mel_features/linear_to_mel_weight_matrixPad8log_mel_features/linear_to_mel_weight_matrix/Maximum:z:0>log_mel_features/linear_to_mel_weight_matrix/paddings:output:0*
T0*
_output_shapes
:	@2.
,log_mel_features/linear_to_mel_weight_matrixż
log_mel_features/MatMulMatMullog_mel_features/Abs:y:05log_mel_features/linear_to_mel_weight_matrix:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
log_mel_features/MatMulu
log_mel_features/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
log_mel_features/add/yŤ
log_mel_features/addAddV2!log_mel_features/MatMul:product:0log_mel_features/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
log_mel_features/add
log_mel_features/LogLoglog_mel_features/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
log_mel_features/Log
#log_mel_features/frame/frame_lengthConst*
_output_shapes
: *
dtype0*
value	B :`2%
#log_mel_features/frame/frame_length
!log_mel_features/frame/frame_stepConst*
_output_shapes
: *
dtype0*
value	B :`2#
!log_mel_features/frame/frame_step|
log_mel_features/frame/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
log_mel_features/frame/axis
log_mel_features/frame/ShapeShapelog_mel_features/Log:y:0*
T0*
_output_shapes
:2
log_mel_features/frame/Shape|
log_mel_features/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :2
log_mel_features/frame/Rank
"log_mel_features/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2$
"log_mel_features/frame/range/start
"log_mel_features/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2$
"log_mel_features/frame/range/deltaá
log_mel_features/frame/rangeRange+log_mel_features/frame/range/start:output:0$log_mel_features/frame/Rank:output:0+log_mel_features/frame/range/delta:output:0*
_output_shapes
:2
log_mel_features/frame/range˘
*log_mel_features/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*log_mel_features/frame/strided_slice/stackŚ
,log_mel_features/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,log_mel_features/frame/strided_slice/stack_1Ś
,log_mel_features/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,log_mel_features/frame/strided_slice/stack_2ě
$log_mel_features/frame/strided_sliceStridedSlice%log_mel_features/frame/range:output:03log_mel_features/frame/strided_slice/stack:output:05log_mel_features/frame/strided_slice/stack_1:output:05log_mel_features/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$log_mel_features/frame/strided_slice~
log_mel_features/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
log_mel_features/frame/sub/y­
log_mel_features/frame/subSub$log_mel_features/frame/Rank:output:0%log_mel_features/frame/sub/y:output:0*
T0*
_output_shapes
: 2
log_mel_features/frame/subł
log_mel_features/frame/sub_1Sublog_mel_features/frame/sub:z:0-log_mel_features/frame/strided_slice:output:0*
T0*
_output_shapes
: 2
log_mel_features/frame/sub_1
log_mel_features/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2!
log_mel_features/frame/packed/1ď
log_mel_features/frame/packedPack-log_mel_features/frame/strided_slice:output:0(log_mel_features/frame/packed/1:output:0 log_mel_features/frame/sub_1:z:0*
N*
T0*
_output_shapes
:2
log_mel_features/frame/packed
&log_mel_features/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&log_mel_features/frame/split/split_dim
log_mel_features/frame/splitSplitV%log_mel_features/frame/Shape:output:0&log_mel_features/frame/packed:output:0/log_mel_features/frame/split/split_dim:output:0*
T0*

Tlen0*$
_output_shapes
: ::*
	num_split2
log_mel_features/frame/split
$log_mel_features/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2&
$log_mel_features/frame/Reshape/shapeÂ
log_mel_features/frame/ReshapeReshape%log_mel_features/frame/split:output:1-log_mel_features/frame/Reshape/shape:output:0*
T0*
_output_shapes
: 2 
log_mel_features/frame/Reshape|
log_mel_features/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B : 2
log_mel_features/frame/Size
log_mel_features/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B :2
log_mel_features/frame/Size_1ť
log_mel_features/frame/sub_2Sub'log_mel_features/frame/Reshape:output:0,log_mel_features/frame/frame_length:output:0*
T0*
_output_shapes
: 2
log_mel_features/frame/sub_2˝
log_mel_features/frame/floordivFloorDiv log_mel_features/frame/sub_2:z:0*log_mel_features/frame/frame_step:output:0*
T0*
_output_shapes
: 2!
log_mel_features/frame/floordiv~
log_mel_features/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :2
log_mel_features/frame/add/xŽ
log_mel_features/frame/addAddV2%log_mel_features/frame/add/x:output:0#log_mel_features/frame/floordiv:z:0*
T0*
_output_shapes
: 2
log_mel_features/frame/add
 log_mel_features/frame/Maximum/xConst*
_output_shapes
: *
dtype0*
value	B : 2"
 log_mel_features/frame/Maximum/xˇ
log_mel_features/frame/MaximumMaximum)log_mel_features/frame/Maximum/x:output:0log_mel_features/frame/add:z:0*
T0*
_output_shapes
: 2 
log_mel_features/frame/Maximum
 log_mel_features/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :`2"
 log_mel_features/frame/gcd/Const
#log_mel_features/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :`2%
#log_mel_features/frame/floordiv_1/yĎ
!log_mel_features/frame/floordiv_1FloorDiv,log_mel_features/frame/frame_length:output:0,log_mel_features/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: 2#
!log_mel_features/frame/floordiv_1
#log_mel_features/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :`2%
#log_mel_features/frame/floordiv_2/yÍ
!log_mel_features/frame/floordiv_2FloorDiv*log_mel_features/frame/frame_step:output:0,log_mel_features/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: 2#
!log_mel_features/frame/floordiv_2
#log_mel_features/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :`2%
#log_mel_features/frame/floordiv_3/yĘ
!log_mel_features/frame/floordiv_3FloorDiv'log_mel_features/frame/Reshape:output:0,log_mel_features/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: 2#
!log_mel_features/frame/floordiv_3~
log_mel_features/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
log_mel_features/frame/mul/yŽ
log_mel_features/frame/mulMul%log_mel_features/frame/floordiv_3:z:0%log_mel_features/frame/mul/y:output:0*
T0*
_output_shapes
: 2
log_mel_features/frame/mulŚ
&log_mel_features/frame/concat/values_1Packlog_mel_features/frame/mul:z:0*
N*
T0*
_output_shapes
:2(
&log_mel_features/frame/concat/values_1
"log_mel_features/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"log_mel_features/frame/concat/axis¤
log_mel_features/frame/concatConcatV2%log_mel_features/frame/split:output:0/log_mel_features/frame/concat/values_1:output:0%log_mel_features/frame/split:output:2+log_mel_features/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:2
log_mel_features/frame/concat
*log_mel_features/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :`2,
*log_mel_features/frame/concat_1/values_1/1ć
(log_mel_features/frame/concat_1/values_1Pack%log_mel_features/frame/floordiv_3:z:03log_mel_features/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:2*
(log_mel_features/frame/concat_1/values_1
$log_mel_features/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$log_mel_features/frame/concat_1/axisŹ
log_mel_features/frame/concat_1ConcatV2%log_mel_features/frame/split:output:01log_mel_features/frame/concat_1/values_1:output:0%log_mel_features/frame/split:output:2-log_mel_features/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
log_mel_features/frame/concat_1
!log_mel_features/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: 2#
!log_mel_features/frame/zeros_like
&log_mel_features/frame/ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:2(
&log_mel_features/frame/ones_like/Shape
&log_mel_features/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&log_mel_features/frame/ones_like/ConstÓ
 log_mel_features/frame/ones_likeFill/log_mel_features/frame/ones_like/Shape:output:0/log_mel_features/frame/ones_like/Const:output:0*
T0*
_output_shapes
:2"
 log_mel_features/frame/ones_likeť
#log_mel_features/frame/StridedSliceStridedSlicelog_mel_features/Log:y:0*log_mel_features/frame/zeros_like:output:0&log_mel_features/frame/concat:output:0)log_mel_features/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2%
#log_mel_features/frame/StridedSliceć
 log_mel_features/frame/Reshape_1Reshape,log_mel_features/frame/StridedSlice:output:0(log_mel_features/frame/concat_1:output:0*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙`˙˙˙˙˙˙˙˙˙2"
 log_mel_features/frame/Reshape_1
$log_mel_features/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2&
$log_mel_features/frame/range_1/start
$log_mel_features/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_features/frame/range_1/deltađ
log_mel_features/frame/range_1Range-log_mel_features/frame/range_1/start:output:0"log_mel_features/frame/Maximum:z:0-log_mel_features/frame/range_1/delta:output:0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
log_mel_features/frame/range_1Á
log_mel_features/frame/mul_1Mul'log_mel_features/frame/range_1:output:0%log_mel_features/frame/floordiv_2:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
log_mel_features/frame/mul_1
(log_mel_features/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(log_mel_features/frame/Reshape_2/shape/1Ý
&log_mel_features/frame/Reshape_2/shapePack"log_mel_features/frame/Maximum:z:01log_mel_features/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&log_mel_features/frame/Reshape_2/shapeÔ
 log_mel_features/frame/Reshape_2Reshape log_mel_features/frame/mul_1:z:0/log_mel_features/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 log_mel_features/frame/Reshape_2
$log_mel_features/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2&
$log_mel_features/frame/range_2/start
$log_mel_features/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2&
$log_mel_features/frame/range_2/deltaę
log_mel_features/frame/range_2Range-log_mel_features/frame/range_2/start:output:0%log_mel_features/frame/floordiv_1:z:0-log_mel_features/frame/range_2/delta:output:0*
_output_shapes
:2 
log_mel_features/frame/range_2
(log_mel_features/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2*
(log_mel_features/frame/Reshape_3/shape/0ŕ
&log_mel_features/frame/Reshape_3/shapePack1log_mel_features/frame/Reshape_3/shape/0:output:0%log_mel_features/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:2(
&log_mel_features/frame/Reshape_3/shapeŇ
 log_mel_features/frame/Reshape_3Reshape'log_mel_features/frame/range_2:output:0/log_mel_features/frame/Reshape_3/shape:output:0*
T0*
_output_shapes

:2"
 log_mel_features/frame/Reshape_3Í
log_mel_features/frame/add_1AddV2)log_mel_features/frame/Reshape_2:output:0)log_mel_features/frame/Reshape_3:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
log_mel_features/frame/add_1°
log_mel_features/frame/GatherV2GatherV2)log_mel_features/frame/Reshape_1:output:0 log_mel_features/frame/add_1:z:0-log_mel_features/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙`˙˙˙˙˙˙˙˙˙2!
log_mel_features/frame/GatherV2Ü
(log_mel_features/frame/concat_2/values_1Pack"log_mel_features/frame/Maximum:z:0,log_mel_features/frame/frame_length:output:0*
N*
T0*
_output_shapes
:2*
(log_mel_features/frame/concat_2/values_1
$log_mel_features/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$log_mel_features/frame/concat_2/axisŹ
log_mel_features/frame/concat_2ConcatV2%log_mel_features/frame/split:output:01log_mel_features/frame/concat_2/values_1:output:0%log_mel_features/frame/split:output:2-log_mel_features/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:2!
log_mel_features/frame/concat_2Ů
 log_mel_features/frame/Reshape_4Reshape(log_mel_features/frame/GatherV2:output:0(log_mel_features/frame/concat_2:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2"
 log_mel_features/frame/Reshape_4
vggish/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"˙˙˙˙`   @      2
vggish/Reshape/shapeŻ
vggish/ReshapeReshape)log_mel_features/frame/Reshape_4:output:0vggish/Reshape/shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`@2
vggish/Reshapeź
"vggish/conv1/Conv2D/ReadVariableOpReadVariableOp+vggish_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"vggish/conv1/Conv2D/ReadVariableOpŰ
vggish/conv1/Conv2DConv2Dvggish/Reshape:output:0*vggish/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`@@*
paddingSAME*
strides
2
vggish/conv1/Conv2Dł
#vggish/conv1/BiasAdd/ReadVariableOpReadVariableOp,vggish_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#vggish/conv1/BiasAdd/ReadVariableOpź
vggish/conv1/BiasAddBiasAddvggish/conv1/Conv2D:output:0+vggish/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`@@2
vggish/conv1/BiasAdd
vggish/conv1/ReluReluvggish/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`@@2
vggish/conv1/ReluÄ
vggish/pool1/MaxPoolMaxPoolvggish/conv1/Relu:activations:0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0 @*
ksize
*
paddingSAME*
strides
2
vggish/pool1/MaxPool˝
"vggish/conv2/Conv2D/ReadVariableOpReadVariableOp+vggish_conv2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"vggish/conv2/Conv2D/ReadVariableOpâ
vggish/conv2/Conv2DConv2Dvggish/pool1/MaxPool:output:0*vggish/conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙0 *
paddingSAME*
strides
2
vggish/conv2/Conv2D´
#vggish/conv2/BiasAdd/ReadVariableOpReadVariableOp,vggish_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#vggish/conv2/BiasAdd/ReadVariableOp˝
vggish/conv2/BiasAddBiasAddvggish/conv2/Conv2D:output:0+vggish/conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙0 2
vggish/conv2/BiasAdd
vggish/conv2/ReluReluvggish/conv2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙0 2
vggish/conv2/ReluĹ
vggish/pool2/MaxPoolMaxPoolvggish/conv2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingSAME*
strides
2
vggish/pool2/MaxPoolÖ
*vggish/conv3/conv3_1/Conv2D/ReadVariableOpReadVariableOp3vggish_conv3_conv3_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*vggish/conv3/conv3_1/Conv2D/ReadVariableOpú
vggish/conv3/conv3_1/Conv2DConv2Dvggish/pool2/MaxPool:output:02vggish/conv3/conv3_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
vggish/conv3/conv3_1/Conv2DĚ
+vggish/conv3/conv3_1/BiasAdd/ReadVariableOpReadVariableOp4vggish_conv3_conv3_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+vggish/conv3/conv3_1/BiasAdd/ReadVariableOpÝ
vggish/conv3/conv3_1/BiasAddBiasAdd$vggish/conv3/conv3_1/Conv2D:output:03vggish/conv3/conv3_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv3/conv3_1/BiasAdd 
vggish/conv3/conv3_1/ReluRelu%vggish/conv3/conv3_1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv3/conv3_1/ReluÖ
*vggish/conv3/conv3_2/Conv2D/ReadVariableOpReadVariableOp3vggish_conv3_conv3_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*vggish/conv3/conv3_2/Conv2D/ReadVariableOp
vggish/conv3/conv3_2/Conv2DConv2D'vggish/conv3/conv3_1/Relu:activations:02vggish/conv3/conv3_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
vggish/conv3/conv3_2/Conv2DĚ
+vggish/conv3/conv3_2/BiasAdd/ReadVariableOpReadVariableOp4vggish_conv3_conv3_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+vggish/conv3/conv3_2/BiasAdd/ReadVariableOpÝ
vggish/conv3/conv3_2/BiasAddBiasAdd$vggish/conv3/conv3_2/Conv2D:output:03vggish/conv3/conv3_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv3/conv3_2/BiasAdd 
vggish/conv3/conv3_2/ReluRelu%vggish/conv3/conv3_2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv3/conv3_2/ReluÍ
vggish/pool3/MaxPoolMaxPool'vggish/conv3/conv3_2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingSAME*
strides
2
vggish/pool3/MaxPoolÖ
*vggish/conv4/conv4_1/Conv2D/ReadVariableOpReadVariableOp3vggish_conv4_conv4_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*vggish/conv4/conv4_1/Conv2D/ReadVariableOpú
vggish/conv4/conv4_1/Conv2DConv2Dvggish/pool3/MaxPool:output:02vggish/conv4/conv4_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
vggish/conv4/conv4_1/Conv2DĚ
+vggish/conv4/conv4_1/BiasAdd/ReadVariableOpReadVariableOp4vggish_conv4_conv4_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+vggish/conv4/conv4_1/BiasAdd/ReadVariableOpÝ
vggish/conv4/conv4_1/BiasAddBiasAdd$vggish/conv4/conv4_1/Conv2D:output:03vggish/conv4/conv4_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv4/conv4_1/BiasAdd 
vggish/conv4/conv4_1/ReluRelu%vggish/conv4/conv4_1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv4/conv4_1/ReluÖ
*vggish/conv4/conv4_2/Conv2D/ReadVariableOpReadVariableOp3vggish_conv4_conv4_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*vggish/conv4/conv4_2/Conv2D/ReadVariableOp
vggish/conv4/conv4_2/Conv2DConv2D'vggish/conv4/conv4_1/Relu:activations:02vggish/conv4/conv4_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
vggish/conv4/conv4_2/Conv2DĚ
+vggish/conv4/conv4_2/BiasAdd/ReadVariableOpReadVariableOp4vggish_conv4_conv4_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+vggish/conv4/conv4_2/BiasAdd/ReadVariableOpÝ
vggish/conv4/conv4_2/BiasAddBiasAdd$vggish/conv4/conv4_2/Conv2D:output:03vggish/conv4/conv4_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv4/conv4_2/BiasAdd 
vggish/conv4/conv4_2/ReluRelu%vggish/conv4/conv4_2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/conv4/conv4_2/ReluÍ
vggish/pool4/MaxPoolMaxPool'vggish/conv4/conv4_2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingSAME*
strides
2
vggish/pool4/MaxPool
vggish/Flatten/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙ 0  2
vggish/Flatten/flatten/ConstÄ
vggish/Flatten/flatten/ReshapeReshapevggish/pool4/MaxPool:output:0%vggish/Flatten/flatten/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`2 
vggish/Flatten/flatten/ReshapeÂ
&vggish/fc1/fc1_1/MatMul/ReadVariableOpReadVariableOp/vggish_fc1_fc1_1_matmul_readvariableop_resource* 
_output_shapes
:
` *
dtype02(
&vggish/fc1/fc1_1/MatMul/ReadVariableOpČ
vggish/fc1/fc1_1/MatMulMatMul'vggish/Flatten/flatten/Reshape:output:0.vggish/fc1/fc1_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 2
vggish/fc1/fc1_1/MatMulŔ
'vggish/fc1/fc1_1/BiasAdd/ReadVariableOpReadVariableOp0vggish_fc1_fc1_1_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02)
'vggish/fc1/fc1_1/BiasAdd/ReadVariableOpĆ
vggish/fc1/fc1_1/BiasAddBiasAdd!vggish/fc1/fc1_1/MatMul:product:0/vggish/fc1/fc1_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 2
vggish/fc1/fc1_1/BiasAdd
vggish/fc1/fc1_1/ReluRelu!vggish/fc1/fc1_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 2
vggish/fc1/fc1_1/ReluÂ
&vggish/fc1/fc1_2/MatMul/ReadVariableOpReadVariableOp/vggish_fc1_fc1_2_matmul_readvariableop_resource* 
_output_shapes
:
  *
dtype02(
&vggish/fc1/fc1_2/MatMul/ReadVariableOpÄ
vggish/fc1/fc1_2/MatMulMatMul#vggish/fc1/fc1_1/Relu:activations:0.vggish/fc1/fc1_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 2
vggish/fc1/fc1_2/MatMulŔ
'vggish/fc1/fc1_2/BiasAdd/ReadVariableOpReadVariableOp0vggish_fc1_fc1_2_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02)
'vggish/fc1/fc1_2/BiasAdd/ReadVariableOpĆ
vggish/fc1/fc1_2/BiasAddBiasAdd!vggish/fc1/fc1_2/MatMul:product:0/vggish/fc1/fc1_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 2
vggish/fc1/fc1_2/BiasAdd
vggish/fc1/fc1_2/ReluRelu!vggish/fc1/fc1_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 2
vggish/fc1/fc1_2/Relu°
 vggish/fc2/MatMul/ReadVariableOpReadVariableOp)vggish_fc2_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype02"
 vggish/fc2/MatMul/ReadVariableOp˛
vggish/fc2/MatMulMatMul#vggish/fc1/fc1_2/Relu:activations:0(vggish/fc2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/fc2/MatMulŽ
!vggish/fc2/BiasAdd/ReadVariableOpReadVariableOp*vggish_fc2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vggish/fc2/BiasAdd/ReadVariableOpŽ
vggish/fc2/BiasAddBiasAddvggish/fc2/MatMul:product:0)vggish/fc2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/fc2/BiasAdd
vggish/embeddingIdentityvggish/fc2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
vggish/embedding"-
vggish_embeddingvggish/embedding:output:0*j
_input_shapesY
W:˙˙˙˙˙˙˙˙˙:::::::::::::::::::) %
#
_output_shapes
:˙˙˙˙˙˙˙˙˙"¸J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ż
\

_variables

signatures
__call__

_vggish_fn"
_generic_user_object
Ś
0
1
2
3
4
5
	6

7
8
9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
"
signature_map
,:*@2vggish/conv1/weights
:@2vggish/conv1/biases
-:+@2vggish/conv2/weights
 :2vggish/conv2/biases
6:42vggish/conv3/conv3_1/weights
(:&2vggish/conv3/conv3_1/biases
6:42vggish/conv3/conv3_2/weights
(:&2vggish/conv3/conv3_2/biases
6:42vggish/conv4/conv4_1/weights
(:&2vggish/conv4/conv4_1/biases
6:42vggish/conv4/conv4_2/weights
(:&2vggish/conv4/conv4_2/biases
*:(
` 2vggish/fc1/fc1_1/weights
$:" 2vggish/fc1/fc1_1/biases
*:(
  2vggish/fc1/fc1_2/weights
$:" 2vggish/fc1/fc1_2/biases
$:"
 2vggish/fc2/weights
:2vggish/fc2/biases
Ű2Ř
__inference___call___611ť
˛
FullArgSpec
args
jself

jwaveform
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘
˙˙˙˙˙˙˙˙˙
$B"
 __inference_wrapped_function_569z
__inference___call___611^	
-˘*
#˘ 

waveform˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙{
 __inference_wrapped_function_569W	
&˘#
˘

0˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙