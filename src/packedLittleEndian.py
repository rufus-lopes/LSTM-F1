import ctypes

class PackedLittleEndianStructure(ctypes.LittleEndianStructure):
    #all data is packed in this structure
    _pack_ = 1
    def __repr__(self):
        fstr_list = []
        for field in self._fields_:
            fname = field[0]
            value = getattr(self, fname)
            if isinstance(
                value, (PackedLittleEndianStructure, int, float, bytes)
            ):
                vstr = repr(value)
            elif isinstance(value, ctypes.Array):
                vstr = "[{}]".format(", ".join(repr(e) for e in value))
            else:
                raise RuntimeError(
                    "Bad value {!r} of type {!r}".format(value, type(value))
                )
            fstr = f"{fname}={vstr}"
            fstr_list.append(fstr)
        return "{}({})".format(self.__class__.__name__, ", ".join(fstr_list))
